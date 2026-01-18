from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
class PodmanNetworkDefaults:

    def __init__(self, module, podman_version):
        self.module = module
        self.version = podman_version
        self.defaults = {'driver': 'bridge', 'internal': False, 'ipv6': False}

    def default_dict(self):
        return self.defaults