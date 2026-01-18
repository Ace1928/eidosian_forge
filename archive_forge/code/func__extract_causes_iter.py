import collections
import copy
import io
import os
import sys
import traceback
from oslo_utils import encodeutils
from oslo_utils import reflection
from taskflow import exceptions as exc
from taskflow.utils import iter_utils
from taskflow.utils import schema_utils as su
@classmethod
def _extract_causes_iter(cls, exc_val):
    seen = [exc_val]
    causes = [exc_val]
    while causes:
        exc_val = causes.pop()
        if exc_val is None:
            continue
        suppress_context = getattr(exc_val, '__suppress_context__', False)
        if suppress_context:
            attr_lookups = ['__cause__']
        else:
            attr_lookups = ['__cause__', '__context__']
        nested_exc_val = None
        for attr_name in attr_lookups:
            attr_val = getattr(exc_val, attr_name, None)
            if attr_val is None:
                continue
            if attr_val not in seen:
                nested_exc_val = attr_val
                break
        if nested_exc_val is not None:
            exc_info = (type(nested_exc_val), nested_exc_val, getattr(nested_exc_val, '__traceback__', None))
            seen.append(nested_exc_val)
            causes.append(nested_exc_val)
            yield cls(exc_info=exc_info)