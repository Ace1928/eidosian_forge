import sys
import math
from datetime import datetime
from sentry_sdk.utils import (
from sentry_sdk._compat import (
from sentry_sdk._types import TYPE_CHECKING
def _serialize_node(obj, is_databag=None, is_request_body=None, should_repr_strings=None, segment=None, remaining_breadth=None, remaining_depth=None):
    if segment is not None:
        path.append(segment)
    try:
        with memo.memoize(obj) as result:
            if result:
                return CYCLE_MARKER
            return _serialize_node_impl(obj, is_databag=is_databag, is_request_body=is_request_body, should_repr_strings=should_repr_strings, remaining_depth=remaining_depth, remaining_breadth=remaining_breadth)
    except BaseException:
        capture_internal_exception(sys.exc_info())
        if is_databag:
            return '<failed to serialize, use init(debug=True) to see error logs>'
        return None
    finally:
        if segment is not None:
            path.pop()
            del meta_stack[len(path) + 1:]