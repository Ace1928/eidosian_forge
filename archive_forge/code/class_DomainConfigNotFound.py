import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class DomainConfigNotFound(NotFound):
    message_format = _('Could not find %(group_or_option)s in domain configuration for domain %(domain_id)s.')