import os
import sys
from oslo_config import cfg
from oslo_db import options as db_options
from glance.common import utils
from glance.db import migration as db_migration
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.db.sqlalchemy import api as db_api
from glance.tests import functional
from glance.tests.utils import depends_on_exe
from glance.tests.utils import execute
from glance.tests.utils import skip_if_disabled
def _check_db(self, expected_exitcode):
    with open(self.conf_filepath, 'w') as conf_file:
        conf_file.write('[DEFAULT]\n')
        conf_file.write(self.connection)
        conf_file.flush()
    cmd = '%s -m glance.cmd.manage --config-file %s db check' % (sys.executable, self.conf_filepath)
    exitcode, out, err = execute(cmd, raise_error=True, expected_exitcode=expected_exitcode)
    return (exitcode, out)