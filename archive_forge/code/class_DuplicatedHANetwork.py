from neutron_lib._i18n import _
from neutron_lib import exceptions
class DuplicatedHANetwork(exceptions.Conflict):
    message = _('Project %(project_id)s already has a HA network.')