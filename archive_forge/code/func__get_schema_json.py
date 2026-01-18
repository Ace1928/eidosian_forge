from __future__ import annotations
import json
import pprint
import warnings
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional
from ._imports import import_item
from .corpus.words import generate_corpus_id
from .json_compat import ValidationError, _validator_for_name, get_current_validator
from .reader import get_version
from .warnings import DuplicateCellId, MissingIDFieldWarning
def _get_schema_json(v, version=None, version_minor=None):
    """
    Gets the json schema from a given imported library and nbformat version.
    """
    if (version, version_minor) in v.nbformat_schema:
        schema_path = str(Path(v.__file__).parent / v.nbformat_schema[version, version_minor])
    elif version_minor > v.nbformat_minor:
        schema_path = str(Path(v.__file__).parent / v.nbformat_schema[None, None])
    else:
        msg = 'Cannot find appropriate nbformat schema file.'
        raise AttributeError(msg)
    with Path(schema_path).open(encoding='utf8') as f:
        schema_json = json.load(f)
    return schema_json