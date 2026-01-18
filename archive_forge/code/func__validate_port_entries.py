from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _validate_port_entries(self, port):
    if port == 'all-other':
        return 0
    if '-' in port:
        parts = port.split('-')
        if len(parts) != 2:
            raise F5ModuleError('The correct format for a port range is X-Y, where X is the startport and Y is the end port.')
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            raise F5ModuleError("The ports in a range must be numbers.You provided '{0}' and '{1}'.".format(parts[0], parts[1]))
        if start == end:
            return start
        if start > end:
            return '{0}-{1}'.format(end, start)
        else:
            return port
    else:
        try:
            return int(port)
        except ValueError:
            raise F5ModuleError('The specified destination port is not a number.')