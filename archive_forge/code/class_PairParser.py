from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class PairParser(Parser, metaclass=abc.ABCMeta):
    """Base class for composite argument parsers consisting of a left and right argument parser, with input separated by a delimiter."""

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        namespace = self.create_namespace()
        state.set_namespace(namespace)
        with state.delimit(self.delimiter, self.required) as boundary:
            choice = self.get_left_parser(state).parse(state)
        if boundary.match:
            self.get_right_parser(choice).parse(state)
        return namespace

    @property
    def required(self) -> bool:
        """True if the delimiter (and thus right parser) is required, otherwise False."""
        return False

    @property
    def delimiter(self) -> str:
        """The delimiter to use between the left and right parser."""
        return PAIR_DELIMITER

    @abc.abstractmethod
    def create_namespace(self) -> t.Any:
        """Create and return a namespace."""

    @abc.abstractmethod
    def get_left_parser(self, state: ParserState) -> Parser:
        """Return the parser for the left side."""

    @abc.abstractmethod
    def get_right_parser(self, choice: t.Any) -> Parser:
        """Return the parser for the right side."""