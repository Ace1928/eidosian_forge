from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _add_argument_wrapper(self: argparse._ActionsContainer, *args: Any, nargs: Union[int, str, Tuple[int], Tuple[int, int], Tuple[int, float], None]=None, choices_provider: Optional[ChoicesProviderFunc]=None, completer: Optional[CompleterFunc]=None, suppress_tab_hint: bool=False, descriptive_header: Optional[str]=None, **kwargs: Any) -> argparse.Action:
    """
    Wrapper around _ActionsContainer.add_argument() which supports more settings used by cmd2

    # Args from original function
    :param self: instance of the _ActionsContainer being added to
    :param args: arguments expected by argparse._ActionsContainer.add_argument

    # Customized arguments from original function
    :param nargs: extends argparse nargs functionality by allowing tuples which specify a range (min, max)
                  to specify a max value with no upper bound, use a 1-item tuple (min,)

    # Added args used by ArgparseCompleter
    :param choices_provider: function that provides choices for this argument
    :param completer: tab completion function that provides choices for this argument
    :param suppress_tab_hint: when ArgparseCompleter has no results to show during tab completion, it displays the
                              current argument's help text as a hint. Set this to True to suppress the hint. If this
                              argument's help text is set to argparse.SUPPRESS, then tab hints will not display
                              regardless of the value passed for suppress_tab_hint. Defaults to False.
    :param descriptive_header: if the provided choices are CompletionItems, then this header will display
                               during tab completion. Defaults to None.

    # Args from original function
    :param kwargs: keyword-arguments recognized by argparse._ActionsContainer.add_argument

    Note: You can only use 1 of the following in your argument:
          choices, choices_provider, completer

          See the header of this file for more information

    :return: the created argument action
    :raises: ValueError on incorrect parameter usage
    """
    choices_callables = [choices_provider, completer]
    num_params_set = len(choices_callables) - choices_callables.count(None)
    if num_params_set > 1:
        err_msg = 'Only one of the following parameters may be used at a time:\nchoices_provider, completer'
        raise ValueError(err_msg)
    nargs_range = None
    if nargs is not None:
        nargs_adjusted: Union[int, str, Tuple[int], Tuple[int, int], Tuple[int, float], None]
        if isinstance(nargs, tuple):
            if len(nargs) == 1:
                nargs = (nargs[0], constants.INFINITY)
            if len(nargs) != 2 or not isinstance(nargs[0], int) or (not (isinstance(nargs[1], int) or nargs[1] == constants.INFINITY)):
                raise ValueError('Ranged values for nargs must be a tuple of 1 or 2 integers')
            if nargs[0] >= nargs[1]:
                raise ValueError('Invalid nargs range. The first value must be less than the second')
            if nargs[0] < 0:
                raise ValueError('Negative numbers are invalid for nargs range')
            nargs_range = nargs
            range_min = nargs_range[0]
            range_max = nargs_range[1]
            if range_min == 0:
                if range_max == 1:
                    nargs_adjusted = argparse.OPTIONAL
                    nargs_range = None
                else:
                    nargs_adjusted = argparse.ZERO_OR_MORE
                    if range_max == constants.INFINITY:
                        nargs_range = None
            elif range_min == 1 and range_max == constants.INFINITY:
                nargs_adjusted = argparse.ONE_OR_MORE
                nargs_range = None
            else:
                nargs_adjusted = argparse.ONE_OR_MORE
        else:
            nargs_adjusted = nargs
        kwargs['nargs'] = nargs_adjusted
    custom_attribs: Dict[str, Any] = {}
    for keyword, value in kwargs.items():
        if keyword in CUSTOM_ACTION_ATTRIBS:
            custom_attribs[keyword] = value
    for keyword in custom_attribs:
        del kwargs[keyword]
    new_arg = orig_actions_container_add_argument(self, *args, **kwargs)
    new_arg.set_nargs_range(nargs_range)
    if choices_provider:
        new_arg.set_choices_provider(choices_provider)
    elif completer:
        new_arg.set_completer(completer)
    new_arg.set_suppress_tab_hint(suppress_tab_hint)
    new_arg.set_descriptive_header(descriptive_header)
    for keyword, value in custom_attribs.items():
        attr_setter = getattr(new_arg, f'set_{keyword}', None)
        if attr_setter is not None:
            attr_setter(value)
    return new_arg