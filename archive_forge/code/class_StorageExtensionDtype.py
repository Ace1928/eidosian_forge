from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import missing as libmissing
from pandas._libs.hashtable import object_hash
from pandas._libs.properties import cache_readonly
from pandas.errors import AbstractMethodError
from pandas.core.dtypes.generic import (
class StorageExtensionDtype(ExtensionDtype):
    """ExtensionDtype that may be backed by more than one implementation."""
    name: str
    _metadata = ('storage',)

    def __init__(self, storage: str | None=None) -> None:
        self.storage = storage

    def __repr__(self) -> str:
        return f'{self.name}[{self.storage}]'

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str) and other == self.name:
            return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

    @property
    def na_value(self) -> libmissing.NAType:
        return libmissing.NA