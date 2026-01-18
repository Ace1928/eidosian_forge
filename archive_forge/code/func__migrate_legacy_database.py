import os
from alembic import command as alembic_api
from alembic import config as alembic_config
from alembic import migration as alembic_migration
from oslo_log import log as logging
import sqlalchemy as sa
from heat.db import api as db_api
def _migrate_legacy_database(engine, connection, config):
    """Check if database is a legacy sqlalchemy-migrate-managed database.

    If it is, migrate it by "stamping" the initial alembic schema.
    """
    if not sa.inspect(engine).has_table('migrate_version'):
        return
    context = alembic_migration.MigrationContext.configure(connection)
    if context.get_current_revision():
        return
    LOG.info('The database is still under sqlalchemy-migrate control; fake applying the initial alembic migration')
    alembic_api.stamp(config, ALEMBIC_INIT_VERSION)