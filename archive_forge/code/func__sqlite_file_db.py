import importlib.machinery
import os
import shutil
import textwrap
from sqlalchemy.testing import config
from sqlalchemy.testing import provision
from . import util as testing_util
from .. import command
from .. import script
from .. import util
from ..script import Script
from ..script import ScriptDirectory
from alembic import context
from alembic import op
from alembic import op
from alembic import op
from alembic import op
from alembic import op
from alembic import op
def _sqlite_file_db(tempname='foo.db', future=False, scope=None, **options):
    dir_ = os.path.join(_get_staging_directory(), 'scripts')
    url = 'sqlite:///%s/%s' % (dir_, tempname)
    if scope and util.sqla_14:
        options['scope'] = scope
    return testing_util.testing_engine(url=url, future=future, options=options)