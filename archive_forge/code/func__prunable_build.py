from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import time
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def _prunable_build(build):
    return build['status']['phase'] in ('Complete', 'Failed', 'Error', 'Cancelled')