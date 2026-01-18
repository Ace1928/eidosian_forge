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
def _setup_stores(self):
    glance_store.register_opts(CONF)
    image_dir = os.path.join(self.test_dir, 'images')
    self.config(group='glance_store', filesystem_store_datadir=image_dir)
    glance_store.create_stores()