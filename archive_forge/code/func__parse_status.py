from __future__ import absolute_import, division, print_function
import time
import re
from collections import namedtuple
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import python_2_unicode_compatible
def _parse_status(self, output, err):
    escaped_monit_services = '|'.join([re.escape(x) for x in MONIT_SERVICES])
    pattern = "(%s) '%s'" % (escaped_monit_services, re.escape(self.process_name))
    if not re.search(pattern, output, re.IGNORECASE):
        return Status.MISSING
    status_val = re.findall('^\\s*status\\s*([\\w\\- ]+)', output, re.MULTILINE)
    if not status_val:
        self.exit_fail('Unable to find process status', stdout=output, stderr=err)
    status_val = status_val[0].strip().upper()
    if ' | ' in status_val:
        status_val = status_val.split(' | ')[0]
    if ' - ' not in status_val:
        status_val = status_val.replace(' ', '_')
        return getattr(Status, status_val)
    else:
        status_val, substatus = status_val.split(' - ')
        action, state = substatus.split()
        if action in ['START', 'INITIALIZING', 'RESTART', 'MONITOR']:
            status = Status.OK
        else:
            status = Status.NOT_MONITORED
        if state == 'pending':
            status = status.pending()
        return status