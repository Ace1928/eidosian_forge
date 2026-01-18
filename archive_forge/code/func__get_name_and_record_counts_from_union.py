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
def _get_name_and_record_counts_from_union(schema: List[Schema]) -> Tuple[int, int]:
    record_type_count = 0
    named_type_count = 0
    for s in schema:
        extracted_type = extract_record_type(s)
        if extracted_type == 'record':
            record_type_count += 1
            named_type_count += 1
        elif extracted_type == 'enum' or extracted_type == 'fixed':
            named_type_count += 1
        elif extracted_type not in AVRO_TYPES:
            named_type_count += 1
            record_type_count += 1
    return (named_type_count, record_type_count)