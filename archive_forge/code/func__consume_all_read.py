import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def _consume_all_read():
    bytes_read = 0
    data = io.StringIO('*' * BYTES)
    reader = utils.LimitingReader(data, BYTES - 1)
    byte = reader.read(1)
    while len(byte) != 0:
        bytes_read += 1
        byte = reader.read(1)