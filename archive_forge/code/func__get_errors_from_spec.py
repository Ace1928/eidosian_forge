import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
def _get_errors_from_spec(spec: Dict[str, Any], schema: Dict[str, Any], rootschema: Optional[Dict[str, Any]]=None) -> ValidationErrorList:
    """Uses the relevant jsonschema validator to validate the passed in spec
    against the schema using the rootschema to resolve references.
    The schema and rootschema themselves are not validated but instead considered
    as valid.
    """
    json_schema_draft_url = _get_json_schema_draft_url(rootschema or schema)
    validator_cls = jsonschema.validators.validator_for({'$schema': json_schema_draft_url})
    validator_kwargs: Dict[str, Any] = {}
    if hasattr(validator_cls, 'FORMAT_CHECKER'):
        validator_kwargs['format_checker'] = validator_cls.FORMAT_CHECKER
    if _use_referencing_library():
        schema = _prepare_references_in_schema(schema)
        validator_kwargs['registry'] = _get_referencing_registry(rootschema or schema, json_schema_draft_url)
    else:
        validator_kwargs['resolver'] = jsonschema.RefResolver.from_schema(rootschema) if rootschema is not None else None
    validator = validator_cls(schema, **validator_kwargs)
    errors = list(validator.iter_errors(spec))
    return errors