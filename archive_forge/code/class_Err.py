from typing import Generic, TypeVar, Union
class Err(Generic[E]):

    def __init__(self, e: E) -> None:
        self._e = e

    def err(self) -> E:
        return self._e