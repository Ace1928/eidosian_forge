import os
import jsonschema
import logging
from typing import List
import json
from ray._private.runtime_env.constants import (
@classmethod
def _load_schemas(cls, schema_paths: List[str]):
    for schema_path in schema_paths:
        try:
            with open(schema_path) as f:
                schema = json.load(f)
        except json.decoder.JSONDecodeError:
            logger.error('Invalid runtime env schema %s, skip it.', schema_path)
            continue
        except OSError:
            logger.error('Cannot open runtime env schema %s, skip it.', schema_path)
            continue
        if 'title' not in schema:
            logger.error('No valid title in runtime env schema %s, skip it.', schema_path)
            continue
        if schema['title'] in cls.schemas:
            logger.error("The 'title' of runtime env schema %s conflicts with %s, skip it.", schema_path, cls.schemas[schema['title']])
            continue
        cls.schemas[schema['title']] = schema