from __future__ import annotations
import abc
import typing as t
from ..argparsing.parsers import (
class TargetsNamespaceParser(NamespaceParser, metaclass=abc.ABCMeta):
    """Base class for controller namespace parsers involving multiple targets."""

    @property
    def option_name(self) -> str:
        """The option name used for this parser."""
        return '--target'

    @property
    def dest(self) -> str:
        """The name of the attribute where the value should be stored."""
        return 'targets'

    @property
    def use_list(self) -> bool:
        """True if the destination is a list, otherwise False."""
        return True