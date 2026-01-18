import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class ApplicationCredentialLimitExceeded(ForbiddenNotSecurity):
    message_format = _('Unable to create additional application credentials, maximum of %(limit)d already exceeded for user.')