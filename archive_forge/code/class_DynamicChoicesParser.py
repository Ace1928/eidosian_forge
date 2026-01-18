from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class DynamicChoicesParser(Parser, metaclass=abc.ABCMeta):
    """Base class for composite argument parsers which use a list of choices that can be generated during completion."""

    def __init__(self, conditions: MatchConditions=MatchConditions.CHOICE) -> None:
        self.conditions = conditions

    @abc.abstractmethod
    def get_choices(self, value: str) -> list[str]:
        """Return a list of valid choices based on the given input value."""

    def no_completion_match(self, value: str) -> CompletionUnavailable:
        """Return an instance of CompletionUnavailable when no match was found for the given value."""
        return CompletionUnavailable()

    def no_choices_available(self, value: str) -> ParserError:
        """Return an instance of ParserError when parsing fails and no choices are available."""
        return ParserError('No choices available.')

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        value = state.read()
        choices = self.get_choices(value)
        if state.mode == ParserMode.PARSE or state.incomplete:
            if self.conditions & MatchConditions.CHOICE and state.match(value, choices):
                return value
            if self.conditions & MatchConditions.ANY and value:
                return value
            if self.conditions & MatchConditions.NOTHING and (not value) and state.current_boundary and (not state.current_boundary.match):
                return value
            if state.mode == ParserMode.PARSE:
                if choices:
                    raise ParserError(f'"{value}" not in: {', '.join(choices)}')
                raise self.no_choices_available(value)
            raise CompletionUnavailable()
        matches = [choice for choice in choices if choice.startswith(value)]
        if not matches:
            raise self.no_completion_match(value)
        continuation = state.current_boundary.delimiters if state.current_boundary and state.current_boundary.required else ''
        raise CompletionSuccess(list_mode=state.mode == ParserMode.LIST, consumed=state.consumed, continuation=continuation, matches=matches)