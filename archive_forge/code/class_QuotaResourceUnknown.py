from oslo_utils import excutils
from neutron_lib._i18n import _
class QuotaResourceUnknown(NotFound):
    message = _('Unknown quota resources %(unknown)s.')