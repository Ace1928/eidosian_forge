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
def _validate_upgrade_order(branch, *, engine=None):
    """Validate the upgrade order of the migration branches.

    This is run before allowing the db_sync command to execute. Ensure the
    expand steps have been run before the contract steps.

    :param branch: The name of the branch that the user is trying to
        upgrade.
    """
    if branch == EXPAND_BRANCH:
        return
    if branch == DATA_MIGRATION_BRANCH:
        return
    config = _find_alembic_conf()
    if engine is None:
        with sql.session_for_read() as session:
            engine = session.get_bind()
    script = alembic_script.ScriptDirectory.from_config(config)
    expand_head = None
    for head in script.get_heads():
        if EXPAND_BRANCH in script.get_revision(head).branch_labels:
            expand_head = head
            break
    with engine.connect() as conn:
        context = alembic_migration.MigrationContext.configure(conn)
        current_heads = context.get_current_heads()
    if expand_head not in current_heads:
        raise db_exception.DBMigrationError('You are attempting to upgrade contract ahead of expand. Please refer to https://docs.openstack.org/keystone/latest/admin/identity-upgrading.html to see the proper steps for rolling upgrades.')