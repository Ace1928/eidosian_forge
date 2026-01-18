from __future__ import annotations
import copy
import datetime as dt
import decimal
import inspect
import json
import typing
import uuid
import warnings
from abc import ABCMeta
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from functools import lru_cache
from marshmallow import base, class_registry, types
from marshmallow import fields as ma_fields
from marshmallow.decorators import (
from marshmallow.error_store import ErrorStore
from marshmallow.exceptions import StringNotCollectionError, ValidationError
from marshmallow.orderedset import OrderedSet
from marshmallow.utils import (
from marshmallow.warnings import RemovedInMarshmallow4Warning
def _run_validator(self, validator_func, output, *, original_data, error_store, many, partial, pass_original, index=None):
    try:
        if pass_original:
            validator_func(output, original_data, partial=partial, many=many)
        else:
            validator_func(output, partial=partial, many=many)
    except ValidationError as err:
        error_store.store_error(err.messages, err.field_name, index=index)