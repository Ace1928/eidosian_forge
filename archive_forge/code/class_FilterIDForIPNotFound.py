from oslo_utils import excutils
from neutron_lib._i18n import _
class FilterIDForIPNotFound(NotFound):
    message = _('Filter ID for IP %(ip)s could not be found.')