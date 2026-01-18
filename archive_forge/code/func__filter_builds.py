from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import time
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def _filter_builds(build):
    config = build['metadata'].get('labels', {}).get('openshift.io/build-config.name')
    return build_config is None or (build_config is not None and config in build_config)