import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class PasswordVerificationError(ForbiddenNotSecurity):
    message_format = _('The password length must be less than or equal to %(size)i. The server could not comply with the request because the password is invalid.')