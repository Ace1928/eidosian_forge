import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class CredentialLimitExceeded(ForbiddenNotSecurity):
    message_format = _('Unable to create additional credentials, maximum of %(limit)d already exceeded for user.')