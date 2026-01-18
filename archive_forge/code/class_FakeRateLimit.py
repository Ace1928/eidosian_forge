import copy
import random
from unittest import mock
import uuid
from cinderclient import api_versions
from openstack.block_storage.v2 import _proxy as block_storage_v2_proxy
from openstack.block_storage.v3 import backup as _backup
from openstack.block_storage.v3 import capabilities as _capabilities
from openstack.block_storage.v3 import stats as _stats
from openstack.block_storage.v3 import volume as _volume
from openstack.image.v2 import _proxy as image_v2_proxy
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class FakeRateLimit(object):
    """Data model that represents a flattened view of a single rate limit."""

    def __init__(self, verb, uri, value, remain, unit, next_available):
        self.verb = verb
        self.uri = uri
        self.value = value
        self.remain = remain
        self.unit = unit
        self.next_available = next_available