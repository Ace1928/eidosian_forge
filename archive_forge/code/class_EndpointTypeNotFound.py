from oslo_utils import encodeutils
from neutronclient._i18n import _
class EndpointTypeNotFound(NeutronClientException):
    message = _('Could not find endpoint type %(type_)s in Service Catalog.')