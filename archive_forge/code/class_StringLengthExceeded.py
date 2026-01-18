import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class StringLengthExceeded(ValidationError):
    message_format = _("String length exceeded. The length of string '%(string)s' exceeds the limit of column %(type)s(CHAR(%(length)d)).")