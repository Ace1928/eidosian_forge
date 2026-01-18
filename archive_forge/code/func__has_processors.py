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
@lru_cache(maxsize=8)
def _has_processors(self, tag) -> bool:
    return bool(self._hooks[tag, True] or self._hooks[tag, False])