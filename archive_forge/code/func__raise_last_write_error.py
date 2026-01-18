from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _raise_last_write_error(write_errors: list[Any]) -> NoReturn:
    error = write_errors[-1]
    if error.get('code') == 11000:
        raise DuplicateKeyError(error.get('errmsg'), 11000, error)
    raise WriteError(error.get('errmsg'), error.get('code'), error)