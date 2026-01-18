from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible.module_utils.common.text.converters import to_native
def _detect_remove_operation(client):
    return client.module.params['state'] == 'remove'