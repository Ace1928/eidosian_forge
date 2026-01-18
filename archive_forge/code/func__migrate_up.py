import os
from alembic import command as alembic_api
from alembic import script as alembic_script
import fixtures
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_log import log as logging
import sqlalchemy
from keystone.common import sql
from keystone.common.sql import upgrades
import keystone.conf
from keystone.tests import unit
import keystone.application_credential.backends.sql  # noqa: F401
import keystone.assignment.backends.sql  # noqa: F401
import keystone.assignment.role_backends.sql_model  # noqa: F401
import keystone.catalog.backends.sql  # noqa: F401
import keystone.credential.backends.sql  # noqa: F401
import keystone.endpoint_policy.backends.sql  # noqa: F401
import keystone.federation.backends.sql  # noqa: F401
import keystone.identity.backends.sql_model  # noqa: F401
import keystone.identity.mapping_backends.sql  # noqa: F401
import keystone.limit.backends.sql  # noqa: F401
import keystone.oauth1.backends.sql  # noqa: F401
import keystone.policy.backends.sql  # noqa: F401
import keystone.resource.backends.sql_model  # noqa: F401
import keystone.resource.config_backends.sql  # noqa: F401
import keystone.revoke.backends.sql  # noqa: F401
import keystone.trust.backends.sql  # noqa: F401
def _migrate_up(self, connection, revision):
    version = revision.revision
    if version == self.init_version:
        alembic_api.upgrade(self.config, version)
        return
    self.assertIsNotNone(getattr(self, '_check_%s' % version, None), 'DB Migration %s does not have a test; you must add one' % version)
    pre_upgrade = getattr(self, '_pre_upgrade_%s' % version, None)
    if pre_upgrade:
        pre_upgrade(connection)
    banned_ops = []
    if version not in self.BANNED_OP_EXCEPTIONS:
        for branch_label in revision.branch_labels:
            banned_ops.extend(self.BANNED_OPS[branch_label])
    if self.FIXTURE.DRIVER == 'sqlite':
        banned_ops = []
    with BannedDBSchemaOperations(banned_ops, version):
        alembic_api.upgrade(self.config, version)
    post_upgrade = getattr(self, '_check_%s' % version, None)
    if post_upgrade:
        post_upgrade(connection)