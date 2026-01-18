import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class InvalidPolicyAssociation(Forbidden):
    message_format = _('Invalid mix of entities for policy association: only Endpoint, Service, or Region+Service allowed. Request was - Endpoint: %(endpoint_id)s, Service: %(service_id)s, Region: %(region_id)s.')