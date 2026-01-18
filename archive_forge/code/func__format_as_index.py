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
def _format_as_index(indices):
    """
    (from jsonschema._utils.format_as_index, copied to avoid relying on private API)

    Construct a single string containing indexing operations for the indices.

    For example, [1, 2, "foo"] -> [1][2]["foo"]
    """
    if not indices:
        return ''
    return '[%s]' % ']['.join((repr(index) for index in indices))