from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import sys
import threading
import time
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.command_lib.meta import generate_cli_trees
from googlecloudsdk.core import module_util
from googlecloudsdk.core.console import console_attr
from prompt_toolkit import completion
import six
class InteractiveCliCompleter(completion.Completer):
    """A prompt_toolkit interactive CLI completer.

  This is the wrapper class for the get_completions() callback that is
  called when characters are added to the default input buffer. It's a bit
  hairy because it maintains state between calls to avoid duplicate work,
  especially for completer calls of unknown cost.

  cli.command_count is a serial number that marks the current command line in
  progress. Some of the cached state is reset when get_completions() detects
  that it has changed.

  Attributes:
    cli: The interactive CLI object.
    coshell: The interactive coshell object.
    debug: The debug object.
    empty: Completion request is on an empty arg if True.
    hidden: Complete hidden commands and flags if True.
    last: The last character before the cursor in the completion request.
    manpage_generator: The unknown command man page generator object.
    module_cache: The completer module path cache object.
    parsed_args: The parsed args namespace passed to completer modules.
    parser: The interactive parser object.
    prefix_completer_command_count: If this is equal to cli.command_count then
      command PREFIX TAB completion is enabled. This completion searches PATH
      for executables matching the current PREFIX token. It's fairly expensive
      and volumninous, so we don't want to do it for every completion event.
    _spinner: Private instance of Spinner used for loading during
      ArgCompleter.
  """

    def __init__(self, cli=None, coshell=None, debug=None, interactive_parser=None, args=None, hidden=False, manpage_generator=True):
        self.arg_cache = CompletionCache(self)
        self.cli = cli
        self.coshell = coshell
        self.debug = debug
        self.hidden = hidden
        self.manpage_generator = manpage_generator
        self.module_cache = {}
        self.parser = interactive_parser
        self.parsed_args = args
        self.empty = False
        self._spinner = None
        self.last = ''
        generate_cli_trees.CliTreeGenerator.MemoizeFailures(True)
        self.reset()

    def reset(self):
        """Resets any cached state for the current command being composed."""
        self.DisableExecutableCompletions()
        if self._spinner:
            self._spinner.Stop()
            self._spinner = None

    def SetSpinner(self, spinner):
        """Sets and Unsets current spinner object."""
        self._spinner = spinner

    def DoExecutableCompletions(self):
        """Returns True if command prefix args should use executable completion."""
        return self.prefix_completer_command_count == self.cli.command_count

    def DisableExecutableCompletions(self):
        """Disables command prefix arg executable completion."""
        self.prefix_completer_command_count = _INVALID_COMMAND_COUNT

    def EnableExecutableCompletions(self):
        """Enables command prefix arg executable completion."""
        self.prefix_completer_command_count = self.cli.command_count

    def IsPrefixArg(self, args):
        """Returns True if the input buffer cursor is in a command prefix arg."""
        return not self.empty and args[-1].token_type == parser.ArgTokenType.PREFIX

    def IsSuppressed(self, info):
        """Returns True if the info for a command, group or flag is hidden."""
        if self.hidden:
            return info.get(parser.LOOKUP_NAME, '').startswith('--no-')
        return info.get(parser.LOOKUP_IS_HIDDEN)

    def get_completions(self, doc, event):
        """Yields the completions for doc.

    Args:
      doc: A Document instance containing the interactive command line to
           complete.
      event: The CompleteEvent that triggered this completion.

    Yields:
      Completion instances for doc.
    """
        self.debug.tabs.count().text('@{}:{}'.format(self.cli.command_count, 'explicit' if event.completion_requested else 'implicit'))
        if not doc.text_before_cursor and event.completion_requested:
            if self.DoExecutableCompletions():
                self.DisableExecutableCompletions()
            else:
                self.EnableExecutableCompletions()
            return
        args = self.parser.ParseCommand(doc.text_before_cursor)
        if not args:
            return
        completers = (self.CommandCompleter, self.FlagCompleter, self.PositionalCompleter, self.InteractiveCompleter)
        if self.IsPrefixArg(args) and (self.DoExecutableCompletions() or event.completion_requested):
            completers = (self.InteractiveCompleter,)
        self.last = doc.text_before_cursor[-1] if doc.text_before_cursor else ''
        self.empty = self.last.isspace()
        self.event = event
        self.debug.last.text(self.last)
        self.debug.tokens.text(_PrettyArgs(args))
        for completer in completers:
            choices, offset = completer(args)
            if choices is None:
                continue
            self.debug.tag(completer.__name__).count().text(len(list(choices)))
            if offset is None:
                for choice in choices:
                    yield choice
            else:
                for choice in sorted(choices):
                    yield completion.Completion(choice, start_position=offset)
            return

    def CommandCompleter(self, args):
        """Returns the command/group completion choices for args or None.

    Args:
      args: The CLI tree parsed command args.

    Returns:
      (choices, offset):
        choices - The list of completion strings or None.
        offset - The completion prefix offset.
    """
        arg = args[-1]
        if arg.value.startswith('-'):
            return (None, 0)
        elif self.IsPrefixArg(args):
            node = self.parser.root
            prefix = arg.value
        elif arg.token_type in (parser.ArgTokenType.COMMAND, parser.ArgTokenType.GROUP) and (not self.empty):
            node = args[-2].tree if len(args) > 1 else self.parser.root
            prefix = arg.value
        elif arg.token_type == parser.ArgTokenType.GROUP:
            if not self.empty:
                return ([], 0)
            node = arg.tree
            prefix = ''
        elif arg.token_type == parser.ArgTokenType.UNKNOWN:
            prefix = arg.value
            if self.manpage_generator and (not prefix) and (len(args) == 2) and args[0].value:
                node = generate_cli_trees.LoadOrGenerate(args[0].value)
                if not node:
                    return (None, 0)
                self.parser.root[parser.LOOKUP_COMMANDS][args[0].value] = node
            elif len(args) > 1 and args[-2].token_type == parser.ArgTokenType.GROUP:
                node = args[-2].tree
            else:
                return (None, 0)
        else:
            return (None, 0)
        choices = [k for k, v in six.iteritems(node[parser.LOOKUP_COMMANDS]) if k.startswith(prefix) and (not self.IsSuppressed(v))]
        if choices:
            return (choices, -len(prefix))
        return (None, 0)

    def ArgCompleter(self, args, arg, value):
        """Returns the flag or positional completion choices for arg or [].

    Args:
      args: The CLI tree parsed command args.
      arg: The flag or positional argument.
      value: The (partial) arg value.

    Returns:
      (choices, offset):
        choices - The list of completion strings or None.
        offset - The completion prefix offset.
    """
        choices = arg.get(parser.LOOKUP_CHOICES)
        if choices:
            return ([v for v in choices if v.startswith(value)], -len(value))
        if not value and (not self.event.completion_requested):
            return ([], 0)
        module_path = arg.get(parser.LOOKUP_COMPLETER)
        if not module_path:
            return ([], 0)
        cache = self.module_cache.get(module_path)
        if not cache:
            cache = ModuleCache(module_util.ImportModule(module_path))
            self.module_cache[module_path] = cache
        prefix = value
        if not isinstance(cache.completer_class, type):
            cache.choices = cache.completer_class(prefix=prefix)
        elif cache.stale < time.time():
            old_dict = self.parsed_args.__dict__
            self.parsed_args.__dict__ = {}
            self.parsed_args.__dict__.update(old_dict)
            self.parsed_args.__dict__.update(_NameSpaceDict(args))
            completer = parser_completer.ArgumentCompleter(cache.completer_class, parsed_args=self.parsed_args)
            with Spinner(self.SetSpinner):
                cache.choices = completer(prefix='')
            self.parsed_args.__dict__ = old_dict
            cache.stale = time.time() + cache.timeout
        if arg.get(parser.LOOKUP_TYPE) == 'list':
            parts = value.split(',')
            prefix = parts[-1]
        if not cache.choices:
            return ([], 0)
        return ([v for v in cache.choices if v.startswith(prefix)], -len(prefix))

    def FlagCompleter(self, args):
        """Returns the flag completion choices for args or None.

    Args:
      args: The CLI tree parsed command args.

    Returns:
      (choices, offset):
        choices - The list of completion strings or None.
        offset - The completion prefix offset.
    """
        arg = args[-1]
        if arg.token_type == parser.ArgTokenType.FLAG_ARG and args[-2].token_type == parser.ArgTokenType.FLAG and (not arg.value and self.last in (' ', '=') or (arg.value and (not self.empty))):
            flag = args[-2].tree
            return self.ArgCompleter(args, flag, arg.value)
        elif arg.token_type == parser.ArgTokenType.FLAG:
            if not self.empty:
                flags = {}
                for a in reversed(args):
                    if a.tree and parser.LOOKUP_FLAGS in a.tree:
                        flags = a.tree[parser.LOOKUP_FLAGS]
                        break
                completions = [k for k, v in six.iteritems(flags) if k != arg.value and k.startswith(arg.value) and (not self.IsSuppressed(v))]
                if completions:
                    completions.append(arg.value)
                    return (completions, -len(arg.value))
            flag = arg.tree
            if flag.get(parser.LOOKUP_TYPE) != 'bool':
                completions, offset = self.ArgCompleter(args, flag, '')
                if not self.empty and self.last != '=':
                    completions = [' ' + c for c in completions]
                return (completions, offset)
        elif arg.value.startswith('-'):
            return ([k for k, v in six.iteritems(arg.tree[parser.LOOKUP_FLAGS]) if k.startswith(arg.value) and (not self.IsSuppressed(v))], -len(arg.value))
        return (None, 0)

    def PositionalCompleter(self, args):
        """Returns the positional completion choices for args or None.

    Args:
      args: The CLI tree parsed command args.

    Returns:
      (choices, offset):
        choices - The list of completion strings or None.
        offset - The completion prefix offset.
    """
        arg = args[-1]
        if arg.token_type == parser.ArgTokenType.POSITIONAL:
            return self.ArgCompleter(args, arg.tree, arg.value)
        return (None, 0)

    def InteractiveCompleter(self, args):
        """Returns the interactive completion choices for args or None.

    Args:
      args: The CLI tree parsed command args.

    Returns:
      (choices, offset):
        choices - The list of completion strings or None.
        offset - The completion prefix offset.
    """
        if self.empty and args[-1].value:
            args = args[:]
            args.append(parser.ArgToken('', parser.ArgTokenType.UNKNOWN, None))
        completions = self.arg_cache.Lookup(args)
        if not completions:
            prefix = self.DoExecutableCompletions() and self.IsPrefixArg(args)
            if not self.event.completion_requested and (not prefix):
                return (None, None)
            command = [arg.value for arg in args]
            with Spinner(self.SetSpinner):
                completions = self.coshell.GetCompletions(command, prefix=prefix)
            self.debug.get.count()
            if not completions:
                return (None, None)
            self.arg_cache.Update(args, completions)
        else:
            self.debug.hit.count()
        last = args[-1].value
        offset = -len(last)
        if False and len(completions) == 1 and completions[0].startswith(last):
            return (completions, offset)
        chop = len(os.path.dirname(last))
        uri_sep = _URI_SEP
        uri_sep_index = completions[0].find(uri_sep)
        if uri_sep_index > 0:
            if not last:
                chop = uri_sep_index + len(uri_sep)
        result = []
        strip_trailing_slash = len(completions) != 1
        for c in completions:
            path_completion = self.MakePathCompletion(c, offset, chop, strip_trailing_slash)
            if path_completion:
                result.append(path_completion)
        return (result, None)

    @classmethod
    def MakePathCompletion(cls, value, offset, chop, strip_trailing_slash=True):
        """Returns the Completion object for a file/uri path completion value.

    Args:
      value: The file/path completion value string.
      offset: The Completion object offset used for dropdown display.
      chop: The minimum number of chars to chop from the dropdown items.
      strip_trailing_slash: Strip trailing '/' if True.

    Returns:
      The Completion object for a file path completion value or None if the
      chopped/stripped value is empty.
    """
        display = value
        if chop:
            display = display[chop:].lstrip('/')
        if not display:
            return None
        if strip_trailing_slash and (not value.endswith(_URI_SEP)):
            value = value.rstrip('/')
        if not value:
            return None
        return completion.Completion(value, display=display, start_position=offset)