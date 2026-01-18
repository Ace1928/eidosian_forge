from abc import ABC, abstractmethod
from typing import Any
class SystemPager(Pager):
    """Uses the pager installed on the system."""

    def _pager(self, content: str) -> Any:
        return __import__('pydoc').pager(content)

    def show(self, content: str) -> None:
        """Use the same pager used by pydoc."""
        self._pager(content)