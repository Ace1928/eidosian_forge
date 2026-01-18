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
def _invoke_schema_validators(self, *, error_store: ErrorStore, pass_many: bool, data, original_data, many: bool, partial: bool | types.StrSequenceOrSet | None, field_errors: bool=False):
    for attr_name in self._hooks[VALIDATES_SCHEMA, pass_many]:
        validator = getattr(self, attr_name)
        validator_kwargs = validator.__marshmallow_hook__[VALIDATES_SCHEMA, pass_many]
        if field_errors and validator_kwargs['skip_on_field_errors']:
            continue
        pass_original = validator_kwargs.get('pass_original', False)
        if many and (not pass_many):
            for idx, (item, orig) in enumerate(zip(data, original_data)):
                self._run_validator(validator, item, original_data=orig, error_store=error_store, many=many, partial=partial, index=idx, pass_original=pass_original)
        else:
            self._run_validator(validator, data, original_data=original_data, error_store=error_store, many=many, pass_original=pass_original, partial=partial)