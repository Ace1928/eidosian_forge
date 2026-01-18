from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Union
import pytest
from xarray import DataArray, Dataset, Variable
class DEnum(Enum):
    dim = 'dim'