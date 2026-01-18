import os
import jsonschema
import logging
from typing import List
import json
from ray._private.runtime_env.constants import (
@classmethod
def _load_default_schemas(cls):
    schema_json_files = list()
    for root, _, files in os.walk(cls.default_schema_path):
        for f in files:
            if f.endswith(RAY_RUNTIME_ENV_PLUGIN_SCHEMA_SUFFIX):
                schema_json_files.append(os.path.join(root, f))
        logger.debug(f'Loading the default runtime env schemas: {schema_json_files}.')
        cls._load_schemas(schema_json_files)