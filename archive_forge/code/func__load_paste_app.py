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
def _load_paste_app(self, name, flavor, conf):
    conf_file_path = os.path.join(self.test_dir, '%s-paste.ini' % name)
    with open(conf_file_path, 'w') as conf_file:
        conf_file.write(conf)
        conf_file.flush()
    return config.load_paste_app(name, flavor=flavor, conf_file=conf_file_path)