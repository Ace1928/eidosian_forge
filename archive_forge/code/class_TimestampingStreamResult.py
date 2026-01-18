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
class TimestampingStreamResult(CopyStreamResult):
    """A StreamResult decorator that assigns a timestamp when none is present.

    This is convenient for ensuring events are timestamped.
    """

    def __init__(self, target):
        super().__init__([target])

    def status(self, *args, **kwargs):
        timestamp = kwargs.pop('timestamp', None)
        if timestamp is None:
            timestamp = datetime.datetime.now(utc)
        super().status(*args, timestamp=timestamp, **kwargs)