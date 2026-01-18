import os
from unittest import mock
import glance_store
from oslo_config import cfg
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context
from glance import housekeeping
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def _store_dir(self, store):
    return os.path.join(self.test_dir, store)