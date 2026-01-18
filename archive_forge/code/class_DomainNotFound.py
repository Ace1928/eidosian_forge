import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class DomainNotFound(NotFound):
    message_format = _('Could not find domain: %(domain_id)s.')