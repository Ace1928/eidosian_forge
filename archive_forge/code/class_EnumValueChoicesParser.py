from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class EnumValueChoicesParser(ChoicesParser):
    """Composite argument parser which relies on a static list of choices derived from the values of an enum."""

    def __init__(self, enum_type: t.Type[enum.Enum], conditions: MatchConditions=MatchConditions.CHOICE) -> None:
        self.enum_type = enum_type
        super().__init__(choices=[str(item.value) for item in enum_type], conditions=conditions)

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        value = super().parse(state)
        return self.enum_type(value)