import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing import Sequence as Sequence_
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
import pyarrow_hotfix  # noqa: F401  # to fix vulnerability on pyarrow<14.0.1
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype
from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import logging
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .image import Image, encode_pil_image
from .translation import Translation, TranslationVariableLanguages
class PandasArrayExtensionDtype(PandasExtensionDtype):
    _metadata = 'value_type'

    def __init__(self, value_type: Union['PandasArrayExtensionDtype', np.dtype]):
        self._value_type = value_type

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]):
        if isinstance(array, pa.ChunkedArray):
            array = array.type.wrap_array(pa.concat_arrays([chunk.storage for chunk in array.chunks]))
        zero_copy_only = _is_zero_copy_only(array.storage.type, unnest=True)
        numpy_arr = array.to_numpy(zero_copy_only=zero_copy_only)
        return PandasArrayExtensionArray(numpy_arr)

    @classmethod
    def construct_array_type(cls):
        return PandasArrayExtensionArray

    @property
    def type(self) -> type:
        return np.ndarray

    @property
    def kind(self) -> str:
        return 'O'

    @property
    def name(self) -> str:
        return f'array[{self.value_type}]'

    @property
    def value_type(self) -> np.dtype:
        return self._value_type