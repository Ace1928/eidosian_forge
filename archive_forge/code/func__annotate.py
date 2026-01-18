import sys
import math
from datetime import datetime
from sentry_sdk.utils import (
from sentry_sdk._compat import (
from sentry_sdk._types import TYPE_CHECKING
def _annotate(**meta):
    while len(meta_stack) <= len(path):
        try:
            segment = path[len(meta_stack) - 1]
            node = meta_stack[-1].setdefault(text_type(segment), {})
        except IndexError:
            node = {}
        meta_stack.append(node)
    meta_stack[-1].setdefault('', {}).update(meta)