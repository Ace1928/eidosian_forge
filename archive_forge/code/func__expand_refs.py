from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def _expand_refs(self, definition):
    if isinstance(definition, list):
        return [self._expand_refs(v) for v in definition]
    if not isinstance(definition, dict):
        return definition
    if '$ref' in definition and isinstance(definition['$ref'], str):
        with self._resolver.resolving(definition['$ref']) as schema:
            return schema
    return {k: self._expand_refs(v) for k, v in definition.items()}