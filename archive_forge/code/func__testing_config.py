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
def _testing_config():
    from alembic.config import Config
    if not os.access(_get_staging_directory(), os.F_OK):
        os.mkdir(_get_staging_directory())
    return Config(os.path.join(_get_staging_directory(), 'test_alembic.ini'))