from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def _validate_domain_scope(self):
    if self.domain_scoped and (not self.roles):
        msg = 'User %(user_id)s has no access to domain %(domain_id)s' % {'user_id': self.user_id, 'domain_id': self.domain_id}
        tr_msg = _('User %(user_id)s has no access to domain %(domain_id)s') % {'user_id': self.user_id, 'domain_id': self.domain_id}
        LOG.debug(msg)
        raise exception.Unauthorized(tr_msg)