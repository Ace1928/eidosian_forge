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
def _pre_upgrade_b4f8b3f584e0(self, connection):
    inspector = sqlalchemy.inspect(connection)
    constraints = inspector.get_unique_constraints('trust')
    self.assertNotIn('duplicate_trust_constraint', {x['name'] for x in constraints})
    all_constraints = []
    for c in constraints:
        all_constraints + c.get('column_names', [])
    not_allowed_constraints = ['trustor_user_id', 'trustee_user_id', 'project_id', 'impersonation', 'expires_at']
    for not_c in not_allowed_constraints:
        self.assertNotIn(not_c, all_constraints)