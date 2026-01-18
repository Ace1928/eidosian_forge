import os
from alembic import command as alembic_api
from alembic import config as alembic_config
from alembic import migration as alembic_migration
from alembic import script as alembic_script
from oslo_db import exception as db_exception
from oslo_log import log as logging
from oslo_utils import fileutils
from keystone.common import sql
import keystone.conf
def _find_alembic_conf():
    """Get the project's alembic configuration.

    :returns: An instance of ``alembic.config.Config``
    """
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'alembic.ini')
    config = alembic_config.Config(os.path.abspath(path))
    config.set_main_option('sqlalchemy.url', CONF.database.connection)
    config.attributes['configure_logger'] = False
    version_paths = [VERSIONS_PATH]
    for release in RELEASES:
        for branch in MIGRATION_BRANCHES:
            version_path = os.path.join(VERSIONS_PATH, release, branch)
            version_paths.append(version_path)
    config.set_main_option('version_locations', ' '.join(version_paths))
    return config