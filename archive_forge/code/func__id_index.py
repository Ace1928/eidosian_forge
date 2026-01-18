from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
@property
def _id_index(self) -> dict[int, T_PandasOrXarrayIndex]:
    if self.__id_index is None:
        self.__id_index = {id(idx): idx for idx in self.get_unique()}
    return self.__id_index