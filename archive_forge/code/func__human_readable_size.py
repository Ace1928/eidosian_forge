import http.client as http
import os
import sys
import urllib.parse as urlparse
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
from webob import exc
from glance.common import config
from glance.common import exception
from glance.common import utils
from glance.i18n import _, _LE, _LI, _LW
def _human_readable_size(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return '%3.1f %s%s' % (num, unit, suffix)
        num /= 1024.0
    return '%.1f %s%s' % (num, 'Yi', suffix)