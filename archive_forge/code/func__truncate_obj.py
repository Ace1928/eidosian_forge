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
def _truncate_obj(obj):
    """Truncate objects for use in validation tracebacks

    Cell and output lists are squashed, as are long strings, lists, and dicts.
    """
    if isinstance(obj, dict):
        truncated_dict = {k: _truncate_obj(v) for k, v in list(obj.items())[:_ITEM_LIMIT]}
        if isinstance(truncated_dict.get('cells'), list):
            truncated_dict['cells'] = ['...%i cells...' % len(obj['cells'])]
        if isinstance(truncated_dict.get('outputs'), list):
            truncated_dict['outputs'] = ['...%i outputs...' % len(obj['outputs'])]
        if len(obj) > _ITEM_LIMIT:
            truncated_dict['...'] = '%i keys truncated' % (len(obj) - _ITEM_LIMIT)
        return truncated_dict
    if isinstance(obj, list):
        truncated_list = [_truncate_obj(item) for item in obj[:_ITEM_LIMIT]]
        if len(obj) > _ITEM_LIMIT:
            truncated_list.append('...%i items truncated...' % (len(obj) - _ITEM_LIMIT))
        return truncated_list
    if isinstance(obj, str):
        truncated_str = obj[:_STR_LIMIT]
        if len(obj) > _STR_LIMIT:
            truncated_str += '...'
        return truncated_str
    return obj