import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_tag_key(self, element, from_tag_api=False):
    if from_tag_api:
        id = findtext(element, 'tagKeyId', TYPES_URN)
        name = findtext(element, 'tagKeyName', TYPES_URN)
    else:
        id = element.get('id')
        name = findtext(element, 'name', TYPES_URN)
    return NttCisTagKey(id=id, name=name, description=findtext(element, 'description', TYPES_URN), value_required=self._str2bool(findtext(element, 'valueRequired', TYPES_URN)), display_on_report=self._str2bool(findtext(element, 'displayOnReport', TYPES_URN)))