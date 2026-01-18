from collections.abc import Iterator
from typing import Iterable
class TrackedIterable(Iterable):

    def __init__(self) -> None:
        super().__init__()
        self.last_item = None

    def __repr__(self) -> str:
        if self.last_item is None:
            super().__repr__()
        else:
            return f'{self.__class__.__name__}(current={self.last_item})'