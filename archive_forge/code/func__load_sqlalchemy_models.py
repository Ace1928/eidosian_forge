import functools
import os
import fixtures
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from keystone.common import sql
import keystone.conf
from keystone.tests import unit
@run_once
def _load_sqlalchemy_models():
    """Find all modules containing SQLAlchemy models and import them.

    This creates more consistent, deterministic test runs because tables
    for all core and extension models are always created in the test
    database. We ensure this by importing all modules that contain model
    definitions.

    The database schema during test runs is created using reflection.
    Reflection is simply SQLAlchemy taking the model definitions for
    all models currently imported and making tables for each of them.
    The database schema created during test runs may vary between tests
    as more models are imported. Importing all models at the start of
    the test run avoids this problem.

    """
    keystone_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    for root, dirs, files in os.walk(keystone_root):
        root = root[len(keystone_root):]
        if root.endswith('backends') and 'sql.py' in files:
            module_root = 'keystone.%s' % root.replace(os.sep, '.').lstrip('.')
            module_components = module_root.split('.')
            module_without_backends = ''
            for x in range(0, len(module_components) - 1):
                module_without_backends += module_components[x] + '.'
            module_without_backends = module_without_backends.rstrip('.')
            module_name = module_root + '.sql'
            __import__(module_name)