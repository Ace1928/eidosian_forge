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
def _create_generator(self, chunk_size, max_iterations):
    chars = b'abc'
    iteration = 0
    while True:
        index = iteration % len(chars)
        chunk = chars[index:index + 1] * chunk_size
        yield chunk
        iteration += 1
        if iteration >= max_iterations:
            return