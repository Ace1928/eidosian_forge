import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class SAMLSigningError(UnexpectedError):
    debug_message_format = _('Unable to sign SAML assertion. It is likely that this server does not have xmlsec1 installed or this is the result of misconfiguration. Reason %(reason)s.')