from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class KeyValueParser(Parser, metaclass=abc.ABCMeta):
    """Base class for key/value composite argument parsers."""

    @abc.abstractmethod
    def get_parsers(self, state: ParserState) -> dict[str, Parser]:
        """Return a dictionary of key names and value parsers."""

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        namespace = state.current_namespace
        parsers = self.get_parsers(state)
        keys = list(parsers)
        with state.delimit(PAIR_DELIMITER, required=False) as pair:
            while pair.ready:
                with state.delimit(ASSIGNMENT_DELIMITER):
                    key = ChoicesParser(keys).parse(state)
                value = parsers[key].parse(state)
                setattr(namespace, key, value)
                keys.remove(key)
        return namespace