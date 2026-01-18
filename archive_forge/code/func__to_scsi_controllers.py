import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_scsi_controllers(self, elements):
    return NttCisScsiController(id=elements.get('id'), adapter_type=elements.get('adapterType'), bus_number=elements.get('busNumber'), state=elements.get('state'))