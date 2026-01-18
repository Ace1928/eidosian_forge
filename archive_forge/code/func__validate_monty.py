from __future__ import annotations
import datetime
import json
import os
import pathlib
import traceback
import types
from collections import OrderedDict, defaultdict
from enum import Enum
from hashlib import sha1
from importlib import import_module
from inspect import getfullargspec
from pathlib import Path
from uuid import UUID
@classmethod
def _validate_monty(cls, __input_value):
    """
        pydantic Validator for MSONable pattern
        """
    if isinstance(__input_value, cls):
        return __input_value
    if isinstance(__input_value, dict):
        try:
            new_obj = MontyDecoder().process_decoded(__input_value)
            if isinstance(new_obj, cls):
                return new_obj
            return cls(**__input_value)
        except Exception:
            raise ValueError(f'Error while deserializing {cls.__name__} object: {traceback.format_exc()}')
    raise ValueError(f'Must provide {cls.__name__}, the as_dict form, or the proper')