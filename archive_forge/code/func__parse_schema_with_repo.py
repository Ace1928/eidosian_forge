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
def _parse_schema_with_repo(schema, repo, named_schemas, write_hint, injected_schemas):
    try:
        schema_copy = deepcopy(named_schemas)
        return parse_schema(schema, named_schemas=named_schemas, _write_hint=write_hint)
    except UnknownType as error:
        missing_subject = error.name
        try:
            sub_schema = _load_schema(missing_subject, repo, named_schemas=schema_copy, write_hint=False, injected_schemas=injected_schemas)
        except SchemaRepositoryError:
            raise error
        if sub_schema['name'] not in injected_schemas:
            injected_schema = _inject_schema(schema, sub_schema)
            if isinstance(schema, str) or isinstance(schema, list):
                schema = injected_schema[0]
            injected_schemas.add(sub_schema['name'])
        return _parse_schema_with_repo(schema, repo, schema_copy, write_hint, injected_schemas)