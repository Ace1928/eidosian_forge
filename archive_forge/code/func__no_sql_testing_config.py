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
def _no_sql_testing_config(dialect='postgresql', directives=''):
    """use a postgresql url with no host so that
    connections guaranteed to fail"""
    dir_ = os.path.join(_get_staging_directory(), 'scripts')
    return _write_config_file('\n[alembic]\nscript_location = %s\nsqlalchemy.url = %s://\n%s\n\n[loggers]\nkeys = root\n\n[handlers]\nkeys = console\n\n[logger_root]\nlevel = WARN\nhandlers = console\nqualname =\n\n[handler_console]\nclass = StreamHandler\nargs = (sys.stderr,)\nlevel = NOTSET\nformatter = generic\n\n[formatters]\nkeys = generic\n\n[formatter_generic]\nformat = %%(levelname)-5.5s [%%(name)s] %%(message)s\ndatefmt = %%H:%%M:%%S\n\n' % (dir_, dialect, directives))