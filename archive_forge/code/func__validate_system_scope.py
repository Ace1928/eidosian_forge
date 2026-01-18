from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def _validate_system_scope(self):
    if self.system_scoped and (not self.roles):
        msg = 'User %(user_id)s has no access to the system' % {'user_id': self.user_id}
        tr_msg = _('User %(user_id)s has no access to the system') % {'user_id': self.user_id}
        LOG.debug(msg)
        raise exception.Unauthorized(tr_msg)