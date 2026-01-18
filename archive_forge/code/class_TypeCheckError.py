from collections import deque
from typing import Deque
class TypeCheckError(Exception):
    """
    Raised by typeguard's type checkers when a type mismatch is detected.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self._path: Deque[str] = deque()

    def append_path_element(self, element: str) -> None:
        self._path.append(element)

    def __str__(self) -> str:
        if self._path:
            return ' of '.join(self._path) + ' ' + str(self.args[0])
        else:
            return str(self.args[0])