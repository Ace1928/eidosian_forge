import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class DomainIdInvalid(ValidationError):
    message_format = _('Domain ID does not conform to required UUID format.')