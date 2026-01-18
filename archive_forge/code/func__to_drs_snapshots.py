import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_drs_snapshots(self, object):
    snapshots = []
    for element in object.findall(fixxpath('snapshot', TYPES_URN)):
        snapshots.append(self._to_process(element))
    return snapshots