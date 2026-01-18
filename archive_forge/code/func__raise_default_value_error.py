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
def _raise_default_value_error(default: Any, schema_type: Any, ignore_default_error: bool):
    if ignore_default_error:
        return
    elif isinstance(schema_type, list):
        text = f'a schema in union with type: {schema_type}'
    else:
        text = f'schema type: {schema_type}'
    raise SchemaParseException(f'Default value <{default}> must match {text}')