from __future__ import annotations
import collections.abc as c
import typing as t
from ...host_configs import (
from ..argparsing.parsers import (
class PlatformParser(ChoicesParser):
    """Composite argument parser for "{platform}/{version}" formatted choices."""

    def __init__(self, choices: list[str]) -> None:
        super().__init__(choices, conditions=MatchConditions.CHOICE | MatchConditions.ANY)

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        value = super().parse(state)
        if len(value.split('/')) != 2:
            raise ParserError(f'invalid platform format: {value}')
        return value