from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
class ChoicesCallable:
    """
    Enables using a callable as the choices provider for an argparse argument.
    While argparse has the built-in choices attribute, it is limited to an iterable.
    """

    def __init__(self, is_completer: bool, to_call: Union[CompleterFunc, ChoicesProviderFunc]) -> None:
        """
        Initializer
        :param is_completer: True if to_call is a tab completion routine which expects
                             the args: text, line, begidx, endidx
        :param to_call: the callable object that will be called to provide choices for the argument
        """
        self.is_completer = is_completer
        if is_completer:
            if not isinstance(to_call, (CompleterFuncBase, CompleterFuncWithTokens)):
                raise ValueError('With is_completer set to true, to_call must be either CompleterFunc, CompleterFuncWithTokens')
        elif not isinstance(to_call, (ChoicesProviderFuncBase, ChoicesProviderFuncWithTokens)):
            raise ValueError('With is_completer set to false, to_call must be either: ChoicesProviderFuncBase, ChoicesProviderFuncWithTokens')
        self.to_call = to_call

    @property
    def completer(self) -> CompleterFunc:
        if not isinstance(self.to_call, (CompleterFuncBase, CompleterFuncWithTokens)):
            raise ValueError('Function is not a CompleterFunc')
        return self.to_call

    @property
    def choices_provider(self) -> ChoicesProviderFunc:
        if not isinstance(self.to_call, (ChoicesProviderFuncBase, ChoicesProviderFuncWithTokens)):
            raise ValueError('Function is not a ChoicesProviderFunc')
        return self.to_call