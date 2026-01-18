from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def _fmt_dt_str(self, cmd, host, duration, author=None, comment=None, start=None, svc=None, fixed=1, trigger=0):
    """
        Format an external-command downtime string.

        cmd - Nagios command ID
        host - Host schedule downtime on
        duration - Minutes to schedule downtime for
        author - Name to file the downtime as
        comment - Reason for running this command (upgrade, reboot, etc)
        start - Start of downtime in seconds since 12:00AM Jan 1 1970
          Default is to use the entry time (now)
        svc - Service to schedule downtime for, omit when for host downtime
        fixed - Start now if 1, start when a problem is detected if 0
        trigger - Optional ID of event to start downtime from. Leave as 0 for
          fixed downtime.

        Syntax: [submitted] COMMAND;<host_name>;[<service_description>]
        <start_time>;<end_time>;<fixed>;<trigger_id>;<duration>;<author>;
        <comment>
        """
    entry_time = self._now()
    if start is None:
        start = entry_time
    hdr = '[%s] %s;%s;' % (entry_time, cmd, host)
    duration_s = duration * 60
    end = start + duration_s
    if not author:
        author = self.author
    if not comment:
        comment = self.comment
    if svc is not None:
        dt_args = [svc, str(start), str(end), str(fixed), str(trigger), str(duration_s), author, comment]
    else:
        dt_args = [str(start), str(end), str(fixed), str(trigger), str(duration_s), author, comment]
    dt_arg_str = ';'.join(dt_args)
    dt_str = hdr + dt_arg_str + '\n'
    return dt_str