from __future__ import annotations
import json
from pathlib import Path, PurePath
from typing import Any, Dict, Union
from jsonschema import FormatChecker, validators
from referencing import Registry
from referencing.jsonschema import DRAFT7
from . import yaml
from .validators import draft7_format_checker, validate_schema
@staticmethod
def _ensure_yaml_loaded(schema: SchemaType, was_str: bool=False) -> None:
    """Ensures schema was correctly loaded into a dictionary. Raises
        EventSchemaLoadingError otherwise."""
    if isinstance(schema, dict):
        return
    error_msg = 'Could not deserialize schema into a dictionary.'

    def intended_as_path(schema: str) -> bool:
        path = Path(schema)
        return path.match('*.yml') or path.match('*.yaml') or path.match('*.json')
    if was_str and intended_as_path(schema):
        error_msg += ' Paths to schema files must be explicitly wrapped in a Pathlib object.'
    else:
        error_msg += ' Double check the schema and ensure it is in the proper form.'
    raise EventSchemaLoadingError(error_msg)