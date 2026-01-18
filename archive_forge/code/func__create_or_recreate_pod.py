from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def _create_or_recreate_pod(self):
    """Ensure pod exists and is exactly as it should be by input params."""
    changed = False
    if self.pod.exists:
        if self.pod.different or self.recreate:
            self.pod.recreate()
            self.results['actions'].append('recreated %s' % self.pod.name)
            changed = True
    elif not self.pod.exists:
        self.pod.create()
        self.results['actions'].append('created %s' % self.pod.name)
        changed = True
    return changed