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
def _invoke_load_processors(self, tag: str, data, *, many: bool, original_data, partial: bool | types.StrSequenceOrSet | None):
    data = self._invoke_processors(tag, pass_many=True, data=data, many=many, original_data=original_data, partial=partial)
    data = self._invoke_processors(tag, pass_many=False, data=data, many=many, original_data=original_data, partial=partial)
    return data