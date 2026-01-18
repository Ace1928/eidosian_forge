import copy
from unittest import mock
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.allocation
def _fake_allocation(self, state, error=None):
    return mock.Mock(state=state, last_error=error)