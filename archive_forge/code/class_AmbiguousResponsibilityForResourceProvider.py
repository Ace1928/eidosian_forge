from neutron_lib._i18n import _
from neutron_lib import exceptions
class AmbiguousResponsibilityForResourceProvider(exceptions.NeutronException):
    """Not clear who's responsible for resource provider."""
    message = _('Expected one driver to be responsible for resource provider %(rsc_provider)s, but got many: %(drivers)s')