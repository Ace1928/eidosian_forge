from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def _strip_metadata(schema: CoreSchema) -> CoreSchema:

    def strip_metadata(s: CoreSchema, recurse: Recurse) -> CoreSchema:
        s = s.copy()
        s.pop('metadata', None)
        if s['type'] == 'model-fields':
            s = s.copy()
            s['fields'] = {k: v.copy() for k, v in s['fields'].items()}
            for field_name, field_schema in s['fields'].items():
                field_schema.pop('metadata', None)
                s['fields'][field_name] = field_schema
            computed_fields = s.get('computed_fields', None)
            if computed_fields:
                s['computed_fields'] = [cf.copy() for cf in computed_fields]
                for cf in computed_fields:
                    cf.pop('metadata', None)
            else:
                s.pop('computed_fields', None)
        elif s['type'] == 'model':
            if s.get('custom_init', True) is False:
                s.pop('custom_init')
            if s.get('root_model', True) is False:
                s.pop('root_model')
            if {'title'}.issuperset(s.get('config', {}).keys()):
                s.pop('config', None)
        return recurse(s, strip_metadata)
    return walk_core_schema(schema, strip_metadata)