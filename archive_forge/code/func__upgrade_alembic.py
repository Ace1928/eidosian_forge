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
def _upgrade_alembic(engine, config, branch):
    revision = 'heads'
    if branch:
        revision = f'{branch}@head'
    with engine.begin() as connection:
        config.attributes['connection'] = connection
        alembic_api.upgrade(config, revision)