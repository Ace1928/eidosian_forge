import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
def _format_meta(metadata, indent):
    """Format the common metadata dictionary in the same manner."""
    if not metadata:
        return []
    lines = ['%s- metadata:' % (' ' * indent)]
    for k, v in metadata.items():
        if k == 'progress' and isinstance(v, misc.NUMERIC_TYPES):
            v = '%0.2f%%' % (v * 100.0)
        lines.append('%s+ %s = %s' % (' ' * (indent + 2), k, v))
    return lines