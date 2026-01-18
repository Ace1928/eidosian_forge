from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _raise_write_concern_error(error: Any) -> NoReturn:
    if _wtimeout_error(error):
        raise WTimeoutError(error.get('errmsg'), error.get('code'), error)
    raise WriteConcernError(error.get('errmsg'), error.get('code'), error)