from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import time
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def _younger_build(build):
    if not self.max_creation_timestamp:
        return False
    creation_timestamp = datetime.strptime(build['metadata']['creationTimestamp'], '%Y-%m-%dT%H:%M:%SZ')
    return creation_timestamp < self.max_creation_timestamp