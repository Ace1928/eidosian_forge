from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def _complete_active_option(self, parser, next_positional, cword_prefix, parsed_args, completions):
    debug('Active actions (L={l}): {a}'.format(l=len(parser.active_actions), a=parser.active_actions))
    isoptional = cword_prefix and cword_prefix[0] in parser.prefix_chars
    greedy_actions = [x for x in parser.active_actions if action_is_greedy(x, isoptional)]
    if greedy_actions:
        assert len(greedy_actions) == 1, 'expect at most 1 greedy action'
        debug('Resetting completions because', greedy_actions[0], 'must consume the next argument')
        self._display_completions = {}
        completions = []
    elif isoptional:
        return completions
    complete_remaining_positionals = False
    for active_action in greedy_actions or parser.active_actions:
        if not active_action.option_strings:
            if action_is_open(active_action):
                complete_remaining_positionals = True
            if not complete_remaining_positionals:
                if action_is_satisfied(active_action) and (not action_is_open(active_action)):
                    debug('Skipping', active_action)
                    continue
        debug('Activating completion for', active_action, active_action._orig_class)
        completer = getattr(active_action, 'completer', None)
        if completer is None:
            if active_action.choices is not None and (not isinstance(active_action, argparse._SubParsersAction)):
                completer = completers.ChoicesCompleter(active_action.choices)
            elif not isinstance(active_action, argparse._SubParsersAction):
                completer = self.default_completer
        if completer:
            if callable(completer):
                completions_from_callable = [c for c in completer(prefix=cword_prefix, action=active_action, parser=parser, parsed_args=parsed_args) if self.validator(c, cword_prefix)]
                if completions_from_callable:
                    completions += completions_from_callable
                    if isinstance(completer, completers.ChoicesCompleter):
                        self._display_completions.update([[x, active_action.help] for x in completions_from_callable])
                    else:
                        self._display_completions.update([[x, ''] for x in completions_from_callable])
            else:
                debug('Completer is not callable, trying the readline completer protocol instead')
                for i in range(9999):
                    next_completion = completer.complete(cword_prefix, i)
                    if next_completion is None:
                        break
                    if self.validator(next_completion, cword_prefix):
                        self._display_completions.update({next_completion: ''})
                        completions.append(next_completion)
            debug('Completions:', completions)
    return completions