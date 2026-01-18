import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def _ensure_key(self, test_id, route_code, timestamp):
    if test_id is None:
        return
    key = (test_id, route_code)
    if key not in self._inprogress:
        self._inprogress[key] = _TestRecord.create(test_id, timestamp)
    return key