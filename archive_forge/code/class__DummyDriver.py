from unittest import mock
from stevedore import enabled
from neutron_lib.tests import _base as base
from neutron_lib.utils import runtime
class _DummyDriver(object):
    driver = mock.sentinel.dummy_driver