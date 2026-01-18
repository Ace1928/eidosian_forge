import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class ResourceDeleteForbidden(ForbiddenNotSecurity):
    message_format = _('Unable to delete immutable %(type)s resource: `%(resource_id)s. Set resource option "immutable" to false first.')