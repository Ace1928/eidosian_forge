from oslo_utils import excutils
from neutron_lib._i18n import _
class InvalidQuotaValue(Conflict):
    message = _('Change would make usage less than 0 for the following resources: %(unders)s.')