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
@staticmethod
def _dict_to_headers(d):
    """Convert a dictionary into one suitable for a HTTP request.

        d: a dictionary

        Returns: the same dictionary, with x-image-meta added to every key
        """
    h = {}
    for key in d:
        if key == 'properties':
            for subkey in d[key]:
                if d[key][subkey] is None:
                    h['x-image-meta-property-%s' % subkey] = ''
                else:
                    h['x-image-meta-property-%s' % subkey] = d[key][subkey]
        else:
            h['x-image-meta-%s' % key] = d[key]
    return h