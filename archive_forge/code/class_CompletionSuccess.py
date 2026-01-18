from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
@dataclasses.dataclass
class CompletionSuccess(Completion):
    """Successful argument completion result."""
    list_mode: bool
    consumed: str
    continuation: str
    matches: list[str] = dataclasses.field(default_factory=list)

    @property
    def preserve(self) -> bool:
        """
        True if argcomplete should not mangle completion values, otherwise False.
        Only used when more than one completion exists to avoid overwriting the word undergoing completion.
        """
        return len(self.matches) > 1 and self.list_mode

    @property
    def completions(self) -> list[str]:
        """List of completion values to return to argcomplete."""
        completions = self.matches
        continuation = '' if self.list_mode else self.continuation
        if not self.preserve:
            completions = [f'{self.consumed}{completion}{continuation}' for completion in completions]
        return completions