import sys
import math
from datetime import datetime
from sentry_sdk.utils import (
from sentry_sdk._compat import (
from sentry_sdk._types import TYPE_CHECKING
def _serialize_node_impl(obj, is_databag, is_request_body, should_repr_strings, remaining_depth, remaining_breadth):
    if isinstance(obj, AnnotatedValue):
        should_repr_strings = False
    if should_repr_strings is None:
        should_repr_strings = _should_repr_strings()
    if is_databag is None:
        is_databag = _is_databag()
    if is_request_body is None:
        is_request_body = _is_request_body()
    if is_databag:
        if is_request_body and keep_request_bodies:
            remaining_depth = float('inf')
            remaining_breadth = float('inf')
        else:
            if remaining_depth is None:
                remaining_depth = MAX_DATABAG_DEPTH
            if remaining_breadth is None:
                remaining_breadth = MAX_DATABAG_BREADTH
    obj = _flatten_annotated(obj)
    if remaining_depth is not None and remaining_depth <= 0:
        _annotate(rem=[['!limit', 'x']])
        if is_databag:
            return _flatten_annotated(strip_string(safe_repr(obj), max_length=max_value_length))
        return None
    if is_databag and global_repr_processors:
        hints = {'memo': memo, 'remaining_depth': remaining_depth}
        for processor in global_repr_processors:
            result = processor(obj, hints)
            if result is not NotImplemented:
                return _flatten_annotated(result)
    sentry_repr = getattr(type(obj), '__sentry_repr__', None)
    if obj is None or isinstance(obj, (bool, number_types)):
        if should_repr_strings or (isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj))):
            return safe_repr(obj)
        else:
            return obj
    elif callable(sentry_repr):
        return sentry_repr(obj)
    elif isinstance(obj, datetime):
        return text_type(format_timestamp(obj)) if not should_repr_strings else safe_repr(obj)
    elif isinstance(obj, Mapping):
        obj = dict(iteritems(obj))
        rv_dict = {}
        i = 0
        for k, v in iteritems(obj):
            if remaining_breadth is not None and i >= remaining_breadth:
                _annotate(len=len(obj))
                break
            str_k = text_type(k)
            v = _serialize_node(v, segment=str_k, should_repr_strings=should_repr_strings, is_databag=is_databag, is_request_body=is_request_body, remaining_depth=remaining_depth - 1 if remaining_depth is not None else None, remaining_breadth=remaining_breadth)
            rv_dict[str_k] = v
            i += 1
        return rv_dict
    elif not isinstance(obj, serializable_str_types) and isinstance(obj, (Set, Sequence)):
        rv_list = []
        for i, v in enumerate(obj):
            if remaining_breadth is not None and i >= remaining_breadth:
                _annotate(len=len(obj))
                break
            rv_list.append(_serialize_node(v, segment=i, should_repr_strings=should_repr_strings, is_databag=is_databag, is_request_body=is_request_body, remaining_depth=remaining_depth - 1 if remaining_depth is not None else None, remaining_breadth=remaining_breadth))
        return rv_list
    if should_repr_strings:
        obj = safe_repr(obj)
    else:
        if isinstance(obj, bytes) or isinstance(obj, bytearray):
            obj = obj.decode('utf-8', 'replace')
        if not isinstance(obj, string_types):
            obj = safe_repr(obj)
    is_span_description = len(path) == 3 and path[0] == 'spans' and (path[-1] == 'description')
    if is_span_description:
        return obj
    return _flatten_annotated(strip_string(obj, max_length=max_value_length))