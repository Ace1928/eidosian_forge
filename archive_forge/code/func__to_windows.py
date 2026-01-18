import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_windows(self, object):
    snapshot_window_elements = object.findall(fixxpath('snapshotWindow', TYPES_URN))
    return [self._to_window(el) for el in snapshot_window_elements]