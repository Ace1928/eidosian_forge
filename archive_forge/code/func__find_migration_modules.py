import importlib
import os.path
import pkgutil
from glance.common import exception
from glance.db import migration as db_migrations
from glance.db.sqlalchemy import api as db_api
def _find_migration_modules(release):
    migrations = list()
    for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
        if module_name.startswith(release):
            migrations.append(module_name)
    migration_modules = list()
    for migration in sorted(migrations):
        module = importlib.import_module('.'.join([__package__, migration]))
        has_migrations_function = getattr(module, 'has_migrations', None)
        migrate_function = getattr(module, 'migrate', None)
        if has_migrations_function is None or migrate_function is None:
            raise exception.InvalidDataMigrationScript(script=module.__name__)
        migration_modules.append(module)
    return migration_modules