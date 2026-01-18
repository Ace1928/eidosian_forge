import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def _fake_node_for_wait(self, state, error=None, target=None):
    spec = ['provision_state', 'last_error', 'target_provision_state']
    return mock.Mock(provision_state=state, last_error=error, target_provision_state=target, spec=spec)