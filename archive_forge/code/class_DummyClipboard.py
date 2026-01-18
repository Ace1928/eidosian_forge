from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Callable
from prompt_toolkit.selection import SelectionType
class DummyClipboard(Clipboard):
    """
    Clipboard implementation that doesn't remember anything.
    """

    def set_data(self, data: ClipboardData) -> None:
        pass

    def set_text(self, text: str) -> None:
        pass

    def rotate(self) -> None:
        pass

    def get_data(self) -> ClipboardData:
        return ClipboardData()