import sys
from os_win._i18n import _
class ClusterGroupMigrationFailed(ClusterException):
    msg_fmt = _('Failed to migrate cluster group %(group_name)s. Expected state %(expected_state)s. Expected owner node: %(expected_node)s. Current group state: %(group_state)s. Current owner node: %(owner_node)s.')