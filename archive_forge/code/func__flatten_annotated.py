import sys
import math
from datetime import datetime
from sentry_sdk.utils import (
from sentry_sdk._compat import (
from sentry_sdk._types import TYPE_CHECKING
def _flatten_annotated(obj):
    if isinstance(obj, AnnotatedValue):
        _annotate(**obj.metadata)
        obj = obj.value
    return obj