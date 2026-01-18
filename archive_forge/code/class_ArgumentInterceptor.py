from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display_info
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core.cache import completion_cache
class ArgumentInterceptor(Argument):
    """ArgumentInterceptor intercepts calls to argparse parsers.

  The argparse module provides no public way to access the arguments that were
  specified on the command line. Argparse itself does the validation when it is
  run from the command line.

  Attributes:
    allow_positional: bool, Whether or not to allow positional arguments.
    defaults: {str:obj}, A dict of {dest: default} for all the arguments added.
    dests: [str], A list of the dests for all arguments.
    flag_args: [argparse.Action], A list of the flag arguments.
    parser: argparse.Parser, The parser whose methods are being intercepted.
    positional_args: [argparse.Action], A list of the positional arguments.
    required: [str], A list of the dests for all required arguments.

  Raises:
    ArgumentException: if a positional argument is made when allow_positional
        is false.
  """

    class ParserData(object):
        """Parser data for the entire command.

    Attributes:
      allow_positional: bool, Allow positional arguments if True.
      ancestor_flag_args: [argparse.Action], The flags for all ancestor groups
        in the cli tree.
      cli_generator: cli.CLILoader, The builder used to generate this CLI.
      command_name: [str], The parts of the command name path.
      concept_handler: calliope.concepts.handlers.RuntimeHandler, a handler
        for concept args.
      defaults: {dest: default}, For all registered arguments.
      dests: [str], A list of the dests for all arguments.
      display_info: [display_info.DisplayInfo], The command display info object.
      flag_args: [ArgumentInterceptor], The flag arguments.
      positional_args: [ArgumentInterceptor], The positional args.
      positional_completers: {Completer}, The set of completers for positionals.
      required: [str], The dests for all required arguments.
    """

        def __init__(self, command_name, cli_generator, allow_positional):
            self.command_name = command_name
            self.cli_generator = cli_generator
            self.allow_positional = allow_positional
            self.ancestor_flag_args = []
            self.concept_handler = None
            self.concepts = None
            self.defaults = {}
            self.dests = []
            self.display_info = display_info.DisplayInfo()
            self.flag_args = []
            self.positional_args = []
            self.positional_completers = set()
            self.required = []

    def __init__(self, parser, cli_generator=None, allow_positional=True, data=None, **kwargs):
        super(ArgumentInterceptor, self).__init__(is_group=True, **kwargs)
        self.is_mutex = kwargs.pop('mutex', False)
        self.help = kwargs.pop('help', None)
        self.parser = parser
        if parser:
            parser.ai = self
        if data:
            self.data = data
        else:
            self.data = ArgumentInterceptor.ParserData(command_name=self.parser._calliope_command.GetPath(), cli_generator=cli_generator, allow_positional=allow_positional)

    @property
    def allow_positional(self):
        return self.data.allow_positional

    @property
    def cli_generator(self):
        return self.data.cli_generator

    @property
    def command_name(self):
        return self.data.command_name

    @property
    def defaults(self):
        return self.data.defaults

    @property
    def display_info(self):
        return self.data.display_info

    @property
    def required(self):
        return self.data.required

    @property
    def dests(self):
        return self.data.dests

    @property
    def positional_args(self):
        return self.data.positional_args

    @property
    def is_hidden(self):
        if self._is_hidden:
            return True
        try:
            next((a for a in self.arguments if not a.is_hidden))
            return False
        except StopIteration:
            flags = []
            for arg in self.arguments:
                if hasattr(arg, 'option_strings'):
                    flags += arg.option_strings
            raise parser_errors.ArgumentException('Groups with arguments and subgroups that are all hidden should be marked hidden.\nCommand: [{}]\nGroup: [{}]\nFlags: [{}]'.format('.'.join(self.command_name), self.help, ', '.join(flags)))

    @property
    def flag_args(self):
        return self.data.flag_args

    @property
    def positional_completers(self):
        return self.data.positional_completers

    @property
    def ancestor_flag_args(self):
        return self.data.ancestor_flag_args

    @property
    def concept_handler(self):
        return self.data.concept_handler

    @property
    def concepts(self):
        return self.data.concepts

    def add_concepts(self, handler):
        from googlecloudsdk.command_lib.concepts import concept_managers
        if isinstance(handler, concept_managers.RuntimeParser):
            self.data.concepts = handler
            return
        if self.data.concept_handler:
            raise AttributeError('It is not permitted to add two runtime handlers to a command class.')
        self.data.concept_handler = handler

    def add_argument(self, *args, **kwargs):
        """add_argument intercepts calls to the parser to track arguments."""
        name = args[0]
        category = kwargs.pop('category', None)
        completer = kwargs.pop('completer', None)
        default = kwargs.get('default')
        dest = kwargs.get('dest')
        if not dest:
            dest = name.lstrip(self.parser.prefix_chars).replace('-', '_')
        do_not_propagate = kwargs.pop('do_not_propagate', False)
        hidden = kwargs.pop('hidden', False) or self._is_hidden
        help_text = kwargs.get('help')
        if not help_text:
            raise ValueError('Argument {} requires help text [hidden={}]'.format(name, hidden))
        if help_text == argparse.SUPPRESS:
            raise ValueError('Argument {} needs hidden=True instead of help=argparse.SUPPRESS.'.format(name))
        require_coverage_in_tests = kwargs.pop('require_coverage_in_tests', True)
        is_replicated = kwargs.pop('is_replicated', False)
        is_global = self.is_global or is_replicated
        nargs = kwargs.get('nargs')
        required = kwargs.get('required', False)
        suggestion_aliases = kwargs.pop('suggestion_aliases', None)
        if suggestion_aliases is None:
            suggestion_aliases = []
        if self.is_global and category == base.COMMONLY_USED_FLAGS:
            category = 'GLOBAL'
        positional = not name.startswith('-')
        if positional:
            if not self.allow_positional:
                raise parser_errors.ArgumentException('Illegal positional argument [{0}] for command [{1}]'.format(name, '.'.join(self.data.command_name)))
            if '-' in name:
                raise parser_errors.ArgumentException("Positional arguments cannot contain a '-'. Illegal argument [{0}] for command [{1}]".format(name, '.'.join(self.data.command_name)))
            if category:
                raise parser_errors.ArgumentException('Positional argument [{0}] cannot have a category in command [{1}]'.format(name, '.'.join(self.data.command_name)))
            if suggestion_aliases:
                raise parser_errors.ArgumentException('Positional argument [{0}] cannot have suggestion aliases in command [{1}]'.format(name, '.'.join(self.data.command_name)))
        self.defaults[dest] = default
        if required:
            self.required.append(dest)
        self.dests.append(dest)
        if positional and 'metavar' not in kwargs:
            kwargs['metavar'] = name.upper()
        if kwargs.get('nargs') is argparse.REMAINDER:
            added_argument = self.parser.AddRemainderArgument(*args, **kwargs)
        else:
            added_argument = self.parser.add_argument(*args, **kwargs)
        self._AttachCompleter(added_argument, completer, positional)
        added_argument.require_coverage_in_tests = require_coverage_in_tests
        added_argument.is_global = is_global
        added_argument.is_group = False
        added_argument.is_hidden = hidden
        added_argument.is_required = required
        added_argument.is_positional = positional
        if hidden:
            added_argument.hidden_help = added_argument.help
            added_argument.help = argparse.SUPPRESS
        if positional:
            if category:
                raise parser_errors.ArgumentException('Positional argument [{0}] cannot have a category in command [{1}]'.format(name, '.'.join(self.data.command_name)))
            if nargs is None or nargs == '+' or (isinstance(nargs, int) and nargs > 0):
                added_argument.is_required = True
            self.positional_args.append(added_argument)
        else:
            if category and required:
                raise parser_errors.ArgumentException('Required flag [{0}] cannot have a category in command [{1}]'.format(name, '.'.join(self.data.command_name)))
            if category == 'REQUIRED':
                raise parser_errors.ArgumentException("Flag [{0}] cannot have category='REQUIRED' in command [{1}]".format(name, '.'.join(self.data.command_name)))
            added_argument.category = category
            added_argument.do_not_propagate = do_not_propagate
            added_argument.is_replicated = is_replicated
            added_argument.suggestion_aliases = suggestion_aliases
            if isinstance(added_argument.choices, dict):
                setattr(added_argument, 'choices_help', added_argument.choices)
                added_argument.choices = sorted(added_argument.choices.keys())
            self.flag_args.append(added_argument)
            inverted_flag = self._AddInvertedBooleanFlagIfNecessary(added_argument, name, dest, kwargs)
            if inverted_flag:
                inverted_flag.category = category
                inverted_flag.do_not_propagate = do_not_propagate
                inverted_flag.is_replicated = is_replicated
                inverted_flag.is_global = is_global
                self.flag_args.append(inverted_flag)
        if not getattr(added_argument, 'is_replicated', False) or len(self.command_name) == 1:
            self.arguments.append(added_argument)
        return added_argument

    def register(self, registry_name, value, object):
        return self.parser.register(registry_name, value, object)

    def set_defaults(self, **kwargs):
        return self.parser.set_defaults(**kwargs)

    def get_default(self, dest):
        return self.parser.get_default(dest)

    def parse_known_args(self, args=None, namespace=None):
        """Hooks ArgumentInterceptor into the argcomplete monkeypatch."""
        return self.parser.parse_known_args(args=args, namespace=namespace)

    def add_group(self, help=None, category=None, mutex=False, required=False, hidden=False, sort_args=True, **kwargs):
        """Adds an argument group with mutex/required attributes to the parser.

    Args:
      help: str, The group help text description.
      category: str, The group flag category name, None for no category.
      mutex: bool, A mutually exclusive group if True.
      required: bool, A required group if True.
      hidden: bool, A hidden group if True.
      sort_args: bool, Whether to sort the group's arguments in help/usage text.
        NOTE - For ordering consistency across gcloud, generally prefer using
        argument categories to organize information (instead of unsetting the
        argument sorting).
      **kwargs: Passed verbatim to ArgumentInterceptor().

    Returns:
      The added argument object.
    """
        if 'description' in kwargs or 'title' in kwargs:
            raise parser_errors.ArgumentException('parser.add_group(): description or title kwargs not supported -- use help=... instead.')
        new_parser = super(type(self.parser), self.parser).add_argument_group()
        group = ArgumentInterceptor(parser=new_parser, is_global=self.is_global, cli_generator=self.cli_generator, allow_positional=self.allow_positional, data=self.data, help=help, category=category, mutex=mutex, required=required, hidden=hidden or self._is_hidden, sort_args=sort_args, **kwargs)
        self.arguments.append(group)
        return group

    def add_argument_group(self, help=None, **kwargs):
        return self.add_group(help=help, **kwargs)

    def add_mutually_exclusive_group(self, help=None, **kwargs):
        return self.add_group(help=help, mutex=True, **kwargs)

    def AddDynamicPositional(self, name, action, **kwargs):
        """Add a positional argument that adds new args on the fly when called.

    Args:
      name: The name/dest of the positional argument.
      action: The argparse Action to use. It must be a subclass of
        parser_extensions.DynamicPositionalAction.
      **kwargs: Passed verbatim to the argparse.ArgumentParser.add_subparsers
        method.

    Returns:
      argparse.Action, The added action.
    """
        kwargs['dest'] = name
        if 'metavar' not in kwargs:
            kwargs['metavar'] = name.upper()
        kwargs['parent_ai'] = self
        action = self.parser.add_subparsers(action=action, **kwargs)
        action.completer = action.Completions
        action.is_group = False
        action.is_hidden = kwargs.get('hidden', False)
        action.is_positional = True
        action.is_required = True
        self.positional_args.append(action)
        self.arguments.append(action)
        return action

    def _FlagArgExists(self, option_string):
        """If flag with the given option_string exists."""
        for action in self.flag_args:
            if option_string in action.option_strings:
                return True
        return False

    def AddFlagActionFromAncestors(self, action):
        """Add a flag action to this parser, but segregate it from the others.

    Segregating the action allows automatically generated help text to ignore
    this flag.

    Args:
      action: argparse.Action, The action for the flag being added.
    """
        for flag in ['--project', '--billing-project', '--universe-domain', '--format']:
            if self._FlagArgExists(flag) and flag in action.option_strings:
                return
        self.parser._add_action(action)
        self.data.ancestor_flag_args.append(action)

    def _AddInvertedBooleanFlagIfNecessary(self, added_argument, name, dest, original_kwargs):
        """Determines whether to create the --no-* flag and adds it to the parser.

    Args:
      added_argument: The argparse argument that was previously created.
      name: str, The name of the flag.
      dest: str, The dest field of the flag.
      original_kwargs: {str: object}, The original set of kwargs passed to the
        ArgumentInterceptor.

    Returns:
      The new argument that was added to the parser or None, if it was not
      necessary to create a new argument.
    """
        action = original_kwargs.get('action')
        wrapped_action = getattr(action, 'wrapped_action', None)
        if wrapped_action is not None:
            action_wrapper = action
            action = wrapped_action
        should_invert, prop = self._ShouldInvertBooleanFlag(name, action)
        if not should_invert:
            return
        default = original_kwargs.get('default', False)
        if prop:
            inverted_synopsis = bool(prop.default)
        elif default not in (True, None):
            inverted_synopsis = False
        elif default:
            inverted_synopsis = bool(default)
        else:
            inverted_synopsis = False
        kwargs = dict(original_kwargs)
        if _IsStoreTrueAction(action):
            action = 'store_false'
        elif _IsStoreFalseAction(action):
            action = 'store_true'
        if wrapped_action is not None:

            class NewAction(action_wrapper):
                pass
            NewAction.SetWrappedAction(action)
            action = NewAction
        kwargs['action'] = action
        if not kwargs.get('dest'):
            kwargs['dest'] = dest
        inverted_argument = self.parser.add_argument(name.replace('--', '--no-', 1), **kwargs)
        if inverted_synopsis:
            setattr(added_argument, 'inverted_synopsis', True)
        inverted_argument.is_hidden = True
        inverted_argument.is_required = added_argument.is_required
        return inverted_argument

    def _ShouldInvertBooleanFlag(self, name, action):
        """Checks if flag name with action is a Boolean flag to invert.

    Args:
      name: str, The flag name.
      action: argparse.Action, The argparse action.

    Returns:
      (False, None) if flag is not a Boolean flag or should not be inverted,
      (True, property) if flag is a Boolean flag associated with a property,
      (False, property) if flag is a non-Boolean flag associated with a property
      otherwise (True, None) if flag is a pure Boolean flag.
    """
        if not name.startswith('--'):
            return (False, None)
        if name.startswith('--no-'):
            return (False, None)
        if '--no-' + name[2:] in self.parser._option_string_actions:
            return (False, None)
        if _IsStoreBoolAction(action):
            return (True, None)
        prop, kind, _ = getattr(action, 'store_property', (None, None, None))
        if prop:
            return (kind == 'bool', prop)
        return (False, None)

    def _AttachCompleter(self, arg, completer, positional):
        """Attaches a completer to arg if one is specified.

    Args:
      arg: The argument to attach the completer to.
      completer: The completer Completer class or argcomplete function object.
      positional: True if argument is a positional.
    """
        from googlecloudsdk.calliope import parser_completer
        if not completer:
            return
        if isinstance(completer, type):
            if positional and issubclass(completer, completion_cache.Completer):
                self.data.positional_completers.add(completer)
            arg.completer = parser_completer.ArgumentCompleter(completer, argument=arg)
        else:
            arg.completer = completer

    def SetSortArgs(self, sort_args):
        """Sets whether or not to sort this group's arguments in help/usage text.

    NOTE - For ordering consistency across gcloud, generally prefer using
    argument categories to organize information (instead of unsetting the
    argument sorting).

    Args:
      sort_args: bool, If arguments in this group should be sorted.
    """
        self._sort_args = sort_args