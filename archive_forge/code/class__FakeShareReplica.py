from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common import constants
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_replicas
class _FakeShareReplica(object):
    id = 'fake_share_replica_id'