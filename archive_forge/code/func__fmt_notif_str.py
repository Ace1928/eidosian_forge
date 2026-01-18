from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def _fmt_notif_str(self, cmd, host=None, svc=None):
    """
        Format an external-command notification string.

        cmd - Nagios command ID.
        host - Host to en/disable notifications on.. A value is not required
          for global downtime
        svc - Service to schedule downtime for. A value is not required
          for host downtime.

        Syntax: [submitted] COMMAND;<host_name>[;<service_description>]
        """
    entry_time = self._now()
    notif_str = '[%s] %s' % (entry_time, cmd)
    if host is not None:
        notif_str += ';%s' % host
        if svc is not None:
            notif_str += ';%s' % svc
    notif_str += '\n'
    return notif_str