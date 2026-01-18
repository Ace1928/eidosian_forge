import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class ImpliedRoleNotFound(NotFound):
    message_format = _('%(prior_role_id)s does not imply %(implied_role_id)s.')