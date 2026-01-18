from __future__ import absolute_import, division, print_function
import re
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _validate_pending_status(self, details):
    """Validate the content of a pending sync operation

        This is a hack. The REST API is not consistent with its 'status' values
        so this method is here to check the returned strings from the operation
        and see if it reported any of these inconsistencies.

        :param details:
        :raises F5ModuleError:
        """
    pattern1 = '.*(?P<msg>Recommended\\s+action.*)'
    for detail in details:
        matches = re.search(pattern1, detail)
        if matches:
            raise F5ModuleError(matches.group('msg'))