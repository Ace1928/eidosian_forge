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
def _setup_database(self):
    sql_connection = 'sqlite:////%s/tests.sqlite' % self.test_dir
    options.set_defaults(CONF, connection=sql_connection)
    glance.db.sqlalchemy.api.clear_db_env()
    glance_db_env = 'GLANCE_DB_TEST_SQLITE_FILE'
    if glance_db_env in os.environ:
        db_location = os.environ[glance_db_env]
        shutil.copyfile(db_location, '%s/tests.sqlite' % self.test_dir)
    else:
        test_utils.db_sync()
        osf, db_location = tempfile.mkstemp()
        os.close(osf)
        shutil.copyfile('%s/tests.sqlite' % self.test_dir, db_location)
        os.environ[glance_db_env] = db_location

        def _delete_cached_db():
            try:
                os.remove(os.environ[glance_db_env])
            except Exception:
                glance_tests.logger.exception('Error cleaning up the file %s' % os.environ[glance_db_env])
        atexit.register(_delete_cached_db)