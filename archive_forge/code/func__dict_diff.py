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
def _dict_diff(a, b):
    """A one way dictionary diff.

    a: a dictionary
    b: a dictionary

    Returns: True if the dictionaries are different
    """
    if set(a.keys()) - set(b.keys()):
        LOG.debug('metadata diff -- source has extra keys: %(keys)s', {'keys': ' '.join(set(a.keys()) - set(b.keys()))})
        return True
    for key in a:
        if str(a[key]) != str(b[key]):
            LOG.debug('metadata diff -- value differs for key %(key)s: source "%(source_value)s" vs target "%(target_value)s"', {'key': key, 'source_value': a[key], 'target_value': b[key]})
            return True
    return False