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
def _get_current_heads(engine, config):
    script = alembic_script.ScriptDirectory.from_config(config)
    with engine.connect() as conn:
        context = alembic_migration.MigrationContext.configure(conn)
        heads = context.get_current_heads()
    heads_map = {}
    for head in heads:
        if CONTRACT_BRANCH in script.get_revision(head).branch_labels:
            heads_map[CONTRACT_BRANCH] = head
        else:
            heads_map[EXPAND_BRANCH] = head
    return heads_map