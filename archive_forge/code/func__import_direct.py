import socket
from unittest import mock
import uuid
from cinderclient.v3 import client as cinderclient
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import strutils
from glance.common import wsgi
from glance.tests import functional
def _import_direct(self, image_id, stores):
    """Do an import of image_id to the given stores."""
    body = {'method': {'name': 'glance-direct'}, 'stores': stores, 'all_stores': False}
    return self.api_post('/v2/images/%s/import' % image_id, json=body)