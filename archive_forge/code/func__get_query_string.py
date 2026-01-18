import logging
import urllib
from oslo_config import cfg
from oslo_utils import encodeutils
import requests
from glance_store import capabilities
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _, _LI
import glance_store.location
def _get_query_string(self):
    if self.query:
        return '?%s' % self.query
    return ''