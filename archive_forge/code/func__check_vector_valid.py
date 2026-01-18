import numbers
from typing import (
from typing_extensions import Self
def _check_vector_valid(self, vector: TVector) -> None:
    if not self._is_valid(vector):
        raise ValueError(f'{vector} is not compatible with linear combination {self}')