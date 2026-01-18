from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Union
import pytest
from xarray import DataArray, Dataset, Variable
class CustomHashable:

    def __init__(self, a: int) -> None:
        self.a = a

    def __hash__(self) -> int:
        return self.a