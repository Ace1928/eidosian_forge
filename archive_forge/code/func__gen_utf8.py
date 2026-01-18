import datetime
import uuid
from hashlib import md5
import random
from string import ascii_letters
from typing import Any, Iterator, Dict, List, cast
from .const import (
from .schema import extract_record_type, extract_logical_type, parse_schema
from .types import Schema, NamedSchemas
from ._schema_common import PRIMITIVES
def _gen_utf8() -> str:
    return ''.join(random.choices(ascii_letters, k=10))