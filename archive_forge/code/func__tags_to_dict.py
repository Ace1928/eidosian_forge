import io
import os
import random
import re
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from datetime import datetime
from functools import wraps, partial
import sentry_sdk
from sentry_sdk._compat import text_type, utc_from_timestamp, iteritems
from sentry_sdk.utils import (
from sentry_sdk.envelope import Envelope, Item
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
def _tags_to_dict(tags):
    rv = {}
    for tag_name, tag_value in tags:
        old_value = rv.get(tag_name)
        if old_value is not None:
            if isinstance(old_value, list):
                old_value.append(tag_value)
            else:
                rv[tag_name] = [old_value, tag_value]
        else:
            rv[tag_name] = tag_value
    return rv