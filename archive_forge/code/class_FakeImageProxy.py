import glance_store
from unittest import mock
from glance.common import exception
from glance.common import store_utils
import glance.location
from glance.tests.unit import base
class FakeImageProxy(object):
    size = None
    context = None
    store_api = mock.Mock()
    store_utils = store_utils