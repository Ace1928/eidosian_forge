from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone
import traceback
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def _younger(obj):
    creation_timestamp = datetime.strptime(obj['metadata']['creationTimestamp'], '%Y-%m-%dT%H:%M:%SZ')
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    age = (now - creation_timestamp).seconds / 60
    return age > self.params['keep_younger_than']