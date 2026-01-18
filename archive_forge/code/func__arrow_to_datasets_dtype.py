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
def _arrow_to_datasets_dtype(arrow_type: pa.DataType) -> str:
    """
    _arrow_to_datasets_dtype takes a pyarrow.DataType and converts it to a datasets string dtype.
    In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`
    """
    if pyarrow.types.is_null(arrow_type):
        return 'null'
    elif pyarrow.types.is_boolean(arrow_type):
        return 'bool'
    elif pyarrow.types.is_int8(arrow_type):
        return 'int8'
    elif pyarrow.types.is_int16(arrow_type):
        return 'int16'
    elif pyarrow.types.is_int32(arrow_type):
        return 'int32'
    elif pyarrow.types.is_int64(arrow_type):
        return 'int64'
    elif pyarrow.types.is_uint8(arrow_type):
        return 'uint8'
    elif pyarrow.types.is_uint16(arrow_type):
        return 'uint16'
    elif pyarrow.types.is_uint32(arrow_type):
        return 'uint32'
    elif pyarrow.types.is_uint64(arrow_type):
        return 'uint64'
    elif pyarrow.types.is_float16(arrow_type):
        return 'float16'
    elif pyarrow.types.is_float32(arrow_type):
        return 'float32'
    elif pyarrow.types.is_float64(arrow_type):
        return 'float64'
    elif pyarrow.types.is_time32(arrow_type):
        return f'time32[{pa.type_for_alias(str(arrow_type)).unit}]'
    elif pyarrow.types.is_time64(arrow_type):
        return f'time64[{pa.type_for_alias(str(arrow_type)).unit}]'
    elif pyarrow.types.is_timestamp(arrow_type):
        if arrow_type.tz is None:
            return f'timestamp[{arrow_type.unit}]'
        elif arrow_type.tz:
            return f'timestamp[{arrow_type.unit}, tz={arrow_type.tz}]'
        else:
            raise ValueError(f'Unexpected timestamp object {arrow_type}.')
    elif pyarrow.types.is_date32(arrow_type):
        return 'date32'
    elif pyarrow.types.is_date64(arrow_type):
        return 'date64'
    elif pyarrow.types.is_duration(arrow_type):
        return f'duration[{arrow_type.unit}]'
    elif pyarrow.types.is_decimal128(arrow_type):
        return f'decimal128({arrow_type.precision}, {arrow_type.scale})'
    elif pyarrow.types.is_decimal256(arrow_type):
        return f'decimal256({arrow_type.precision}, {arrow_type.scale})'
    elif pyarrow.types.is_binary(arrow_type):
        return 'binary'
    elif pyarrow.types.is_large_binary(arrow_type):
        return 'large_binary'
    elif pyarrow.types.is_string(arrow_type):
        return 'string'
    elif pyarrow.types.is_large_string(arrow_type):
        return 'large_string'
    else:
        raise ValueError(f'Arrow type {arrow_type} does not have a datasets dtype equivalent.')