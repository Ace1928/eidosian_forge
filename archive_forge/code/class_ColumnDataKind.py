from __future__ import annotations
import json
from enum import Enum
from typing import TYPE_CHECKING, Dict, Final, Literal, Mapping, Union
from typing_extensions import TypeAlias
from streamlit.elements.lib.column_types import ColumnConfig, ColumnType
from streamlit.elements.lib.dicttools import remove_none_values
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.type_util import DataFormat, is_colum_type_arrow_incompatible
class ColumnDataKind(str, Enum):
    INTEGER = 'integer'
    FLOAT = 'float'
    DATE = 'date'
    TIME = 'time'
    DATETIME = 'datetime'
    BOOLEAN = 'boolean'
    STRING = 'string'
    TIMEDELTA = 'timedelta'
    PERIOD = 'period'
    INTERVAL = 'interval'
    BYTES = 'bytes'
    DECIMAL = 'decimal'
    COMPLEX = 'complex'
    LIST = 'list'
    DICT = 'dict'
    EMPTY = 'empty'
    UNKNOWN = 'unknown'