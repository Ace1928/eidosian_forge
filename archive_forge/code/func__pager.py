from abc import ABC, abstractmethod
from typing import Any
def _pager(self, content: str) -> Any:
    return __import__('pydoc').pager(content)