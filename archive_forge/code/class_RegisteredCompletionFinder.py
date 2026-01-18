from __future__ import annotations
import abc
import argparse
import os
import re
import typing as t
from .argcompletion import (
from .parsers import (
class RegisteredCompletionFinder(OptionCompletionFinder):
    """
    Custom option completion finder for argcomplete which allows completion results to be registered.
    These registered completions, if provided, are used to filter the final completion results.
    This works around a known bug: https://github.com/kislyuk/argcomplete/issues/221
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.registered_completions: t.Optional[list[str]] = None

    def completer(self, prefix: str, action: argparse.Action, parsed_args: argparse.Namespace, **kwargs) -> list[str]:
        """
        Return a list of completions for the specified prefix and action.
        Use this as the completer function for argcomplete.
        """
        kwargs.clear()
        del kwargs
        completions = self.get_completions(prefix, action, parsed_args)
        if action.nargs and (not isinstance(action.nargs, int)):
            self.registered_completions = completions
        return completions

    @abc.abstractmethod
    def get_completions(self, prefix: str, action: argparse.Action, parsed_args: argparse.Namespace) -> list[str]:
        """
        Return a list of completions for the specified prefix and action.
        Called by the complete function.
        """

    def quote_completions(self, completions, cword_prequote, last_wordbreak_pos):
        """Modify completion results before returning them."""
        if self.registered_completions is not None:
            allowed_completions = set(self.registered_completions)
            completions = [completion for completion in completions if completion in allowed_completions]
        return super().quote_completions(completions, cword_prequote, last_wordbreak_pos)