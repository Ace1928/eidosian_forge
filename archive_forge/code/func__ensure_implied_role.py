import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _ensure_implied_role(self, prior_role_id, implied_role_id):
    try:
        PROVIDERS.role_api.create_implied_role(prior_role_id, implied_role_id)
        LOG.info('Created implied role where %s implies %s', prior_role_id, implied_role_id)
    except exception.Conflict:
        LOG.info('Implied role where %s implies %s exists, skipping creation.', prior_role_id, implied_role_id)