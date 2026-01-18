import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_window(self, element):
    return {'id': element.get('id'), 'day_of_week': element.get('dayOfWeek'), 'start_hour': element.get('startHour'), 'availability_status': element.get('availabilityStatus')}