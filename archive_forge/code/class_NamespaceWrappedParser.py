from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class NamespaceWrappedParser(NamespaceParser):
    """Composite argument parser that wraps a non-namespace parser and stores the result in a namespace."""

    def __init__(self, dest: str, parser: Parser) -> None:
        self._dest = dest
        self.parser = parser

    def get_value(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result, without storing the result in the namespace."""
        return self.parser.parse(state)

    @property
    def dest(self) -> str:
        """The name of the attribute where the value should be stored."""
        return self._dest