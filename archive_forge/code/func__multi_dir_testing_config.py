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
def _multi_dir_testing_config(sourceless=False, extra_version_location=''):
    dir_ = os.path.join(_get_staging_directory(), 'scripts')
    sqlalchemy_future = 'future' in config.db.__class__.__module__
    url = 'sqlite:///%s/foo.db' % dir_
    return _write_config_file('\n[alembic]\nscript_location = %s\nsqlalchemy.url = %s\nsqlalchemy.future = %s\nsourceless = %s\nversion_locations = %%(here)s/model1/ %%(here)s/model2/ %%(here)s/model3/ %s\n\n[loggers]\nkeys = root\n\n[handlers]\nkeys = console\n\n[logger_root]\nlevel = WARN\nhandlers = console\nqualname =\n\n[handler_console]\nclass = StreamHandler\nargs = (sys.stderr,)\nlevel = NOTSET\nformatter = generic\n\n[formatters]\nkeys = generic\n\n[formatter_generic]\nformat = %%(levelname)-5.5s [%%(name)s] %%(message)s\ndatefmt = %%H:%%M:%%S\n    ' % (dir_, url, 'true' if sqlalchemy_future else 'false', 'true' if sourceless else 'false', extra_version_location))