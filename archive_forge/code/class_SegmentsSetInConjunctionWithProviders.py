from neutron_lib._i18n import _
from neutron_lib import exceptions
class SegmentsSetInConjunctionWithProviders(exceptions.InvalidInput):
    message = _('Segments and provider values cannot both be set.')