from __future__ import annotations
import abc
import typing as t
from ..argparsing.parsers import (
class ControllerNamespaceParser(NamespaceParser, metaclass=abc.ABCMeta):
    """Base class for controller namespace parsers."""

    @property
    def dest(self) -> str:
        """The name of the attribute where the value should be stored."""
        return 'controller'

    def parse(self, state: ParserState) -> t.Any:
        """Parse the input from the given state and return the result."""
        if state.root_namespace.targets:
            raise ControllerRequiredFirstError()
        return super().parse(state)