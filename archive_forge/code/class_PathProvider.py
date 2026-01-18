from __future__ import annotations
import abc
import os
import typing as t
from ..util import (
class PathProvider(metaclass=abc.ABCMeta):
    """Base class for provider plugins that are path based."""
    sequence = 500
    priority = 500

    def __init__(self, root: str) -> None:
        self.root = root

    @staticmethod
    @abc.abstractmethod
    def is_content_root(path: str) -> bool:
        """Return True if the given path is a content root for this provider."""