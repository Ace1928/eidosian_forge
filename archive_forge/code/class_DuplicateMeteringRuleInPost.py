from neutron_lib._i18n import _
from neutron_lib import exceptions
class DuplicateMeteringRuleInPost(exceptions.InUse):
    message = _('Duplicate Metering Rule in POST.')