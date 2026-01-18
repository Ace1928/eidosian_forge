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
def _default_matches_schema(default: Any, schema: Schema) -> bool:
    if schema == 'null' and default is not None or (schema == 'boolean' and (not isinstance(default, bool))) or (schema == 'string' and (not isinstance(default, str))) or (schema == 'bytes' and (not isinstance(default, str))) or (schema == 'double' and (not isinstance(_maybe_float(default), float))) or (schema == 'float' and (not isinstance(_maybe_float(default), float))) or (schema == 'int' and (not isinstance(default, int))) or (schema == 'long' and (not isinstance(default, int))):
        return False
    return True