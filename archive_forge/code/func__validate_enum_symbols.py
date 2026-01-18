import hashlib
from io import StringIO
import math
from os import path
from copy import deepcopy
import re
from typing import Tuple, Set, Optional, List, Any
from .types import DictSchema, Schema, NamedSchemas
from .repository import (
from .const import AVRO_TYPES
from ._schema_common import (
def _validate_enum_symbols(schema):
    symbols = schema['symbols']
    for symbol in symbols:
        if not isinstance(symbol, str) or not SYMBOL_REGEX.fullmatch(symbol):
            raise SchemaParseException('Every symbol must match the regular expression [A-Za-z_][A-Za-z0-9_]*')
    if len(symbols) != len(set(symbols)):
        raise SchemaParseException('All symbols in an enum must be unique')
    if 'default' in schema and schema['default'] not in symbols:
        raise SchemaParseException('Default value for enum must be in symbols list')