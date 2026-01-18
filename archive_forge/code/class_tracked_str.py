from collections.abc import Iterator
from typing import Iterable
class tracked_str(str):
    origins = {}

    def set_origin(self, origin: str):
        if super().__repr__() not in self.origins:
            self.origins[super().__repr__()] = origin

    def get_origin(self):
        return self.origins.get(super().__repr__(), str(self))

    def __repr__(self) -> str:
        if super().__repr__() not in self.origins or self.origins[super().__repr__()] == self:
            return super().__repr__()
        else:
            return f'{str(self)} (origin={self.origins[super().__repr__()]})'