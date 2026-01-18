import argparse
import inspect
import numbers
from collections import (
from typing import (
from .ansi import (
from .constants import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .table_creator import (
def _complete_arg(self, text: str, line: str, begidx: int, endidx: int, arg_state: _ArgumentState, consumed_arg_values: Dict[str, List[str]], *, cmd_set: Optional[CommandSet]=None) -> List[str]:
    """
        Tab completion routine for an argparse argument
        :return: list of completions
        :raises: CompletionError if the completer or choices function this calls raises one
        """
    arg_choices: Union[List[str], ChoicesCallable]
    if arg_state.action.choices is not None:
        arg_choices = list(arg_state.action.choices)
        if not arg_choices:
            return []
        if all((isinstance(x, numbers.Number) for x in arg_choices)):
            arg_choices.sort()
            self._cmd2_app.matches_sorted = True
        for index, choice in enumerate(arg_choices):
            if not isinstance(choice, str):
                arg_choices[index] = str(choice)
    else:
        choices_attr = arg_state.action.get_choices_callable()
        if choices_attr is None:
            return []
        arg_choices = choices_attr
    args = []
    kwargs = {}
    if isinstance(arg_choices, ChoicesCallable):
        self_arg = self._cmd2_app._resolve_func_self(arg_choices.to_call, cmd_set)
        if self_arg is None:
            raise CompletionError('Could not find CommandSet instance matching defining type for completer')
        args.append(self_arg)
        to_call_params = inspect.signature(arg_choices.to_call).parameters
        if ARG_TOKENS in to_call_params:
            arg_tokens = {**self._parent_tokens, **consumed_arg_values}
            arg_tokens.setdefault(arg_state.action.dest, [])
            arg_tokens[arg_state.action.dest].append(text)
            kwargs[ARG_TOKENS] = arg_tokens
    if isinstance(arg_choices, ChoicesCallable) and arg_choices.is_completer:
        args.extend([text, line, begidx, endidx])
        results = arg_choices.completer(*args, **kwargs)
    else:
        completion_items: List[str] = []
        if isinstance(arg_choices, ChoicesCallable):
            if not arg_choices.is_completer:
                choices_func = arg_choices.choices_provider
                if isinstance(choices_func, ChoicesProviderFuncWithTokens):
                    completion_items = choices_func(*args, **kwargs)
                else:
                    completion_items = choices_func(*args)
        else:
            completion_items = arg_choices
        used_values = consumed_arg_values.get(arg_state.action.dest, [])
        completion_items = [choice for choice in completion_items if choice not in used_values]
        results = self._cmd2_app.basic_complete(text, line, begidx, endidx, completion_items)
    if not results:
        self._cmd2_app.matches_sorted = False
        return []
    return self._format_completions(arg_state, results)