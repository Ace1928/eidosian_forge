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
def _relax_additional_properties(obj):
    """relax any `additionalProperties`"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            value = True if key == 'additionalProperties' else _relax_additional_properties(value)
            obj[key] = value
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = _relax_additional_properties(value)
    return obj