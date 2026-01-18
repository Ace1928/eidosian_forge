from __future__ import annotations
from dataclasses import dataclass
import os
import abc
import typing as T
class MesonException(Exception):
    """Exceptions thrown by Meson"""

    def __init__(self, *args: object, file: T.Optional[str]=None, lineno: T.Optional[int]=None, colno: T.Optional[int]=None):
        super().__init__(*args)
        self.file = file
        self.lineno = lineno
        self.colno = colno

    @classmethod
    def from_node(cls, *args: object, node: BaseNode) -> MesonException:
        """Create a MesonException with location data from a BaseNode

        :param node: A BaseNode to set location data from
        :return: A Meson Exception instance
        """
        return cls(*args, file=node.filename, lineno=node.lineno, colno=node.colno)