from oslo_utils import excutils
from neutron_lib._i18n import _
class ExtensionsNotFound(NotFound):
    message = _('Extensions not found: %(extensions)s.')