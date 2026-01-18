import atexit
import os.path
import shutil
import tempfile
import fixtures
import glance_store
from oslo_config import cfg
from oslo_db import options
import glance.common.client
from glance.common import config
import glance.db.sqlalchemy.api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
def _setup_property_protection(self):
    self._copy_data_file('property-protections.conf', self.test_dir)
    self.property_file = os.path.join(self.test_dir, 'property-protections.conf')