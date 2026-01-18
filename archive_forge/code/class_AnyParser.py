from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class AnyParser(ChoicesParser):
    """Composite argument parser which accepts any input value."""

    def __init__(self, nothing: bool=False, no_match_message: t.Optional[str]=None) -> None:
        self.no_match_message = no_match_message
        conditions = MatchConditions.ANY
        if nothing:
            conditions |= MatchConditions.NOTHING
        super().__init__([], conditions=conditions)

    def no_completion_match(self, value: str) -> CompletionUnavailable:
        """Return an instance of CompletionUnavailable when no match was found for the given value."""
        if self.no_match_message:
            return CompletionUnavailable(message=self.no_match_message)
        return super().no_completion_match(value)

    def no_choices_available(self, value: str) -> ParserError:
        """Return an instance of ParserError when parsing fails and no choices are available."""
        if self.no_match_message:
            return ParserError(self.no_match_message)
        return super().no_choices_available(value)