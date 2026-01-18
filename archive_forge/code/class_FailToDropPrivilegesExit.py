from oslo_utils import excutils
from neutron_lib._i18n import _
class FailToDropPrivilegesExit(SystemExit):
    """Exit exception raised when a drop privileges action fails."""
    code = 99