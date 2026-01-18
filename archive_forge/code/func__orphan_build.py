from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import time
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def _orphan_build(build):
    if not _prunable_build(build):
        return False
    config = build['status'].get('config', None)
    if not config:
        return True
    build_config = self.get_build_config(config['name'], config['namespace'])
    return len(build_config) == 0