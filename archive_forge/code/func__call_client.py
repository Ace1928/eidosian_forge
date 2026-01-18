from __future__ import (absolute_import, division, print_function)
import os
import os.path
from ansible.errors import AnsibleFileNotFound, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.copy import (
from ansible_collections.community.docker.plugins.plugin_utils.socket_handler import (
from ansible_collections.community.docker.plugins.plugin_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, DockerException, NotFound
def _call_client(self, callable, not_found_can_be_resource=False):
    try:
        return callable()
    except NotFound as e:
        if not_found_can_be_resource:
            raise AnsibleConnectionFailure('Could not find container "{1}" or resource in it ({0})'.format(e, self.get_option('remote_addr')))
        else:
            raise AnsibleConnectionFailure('Could not find container "{1}" ({0})'.format(e, self.get_option('remote_addr')))
    except APIError as e:
        if e.response is not None and e.response.status_code == 409:
            raise AnsibleConnectionFailure('The container "{1}" has been paused ({0})'.format(e, self.get_option('remote_addr')))
        self.client.fail('An unexpected Docker error occurred for container "{1}": {0}'.format(e, self.get_option('remote_addr')))
    except DockerException as e:
        self.client.fail('An unexpected Docker error occurred for container "{1}": {0}'.format(e, self.get_option('remote_addr')))
    except RequestException as e:
        self.client.fail('An unexpected requests error occurred for container "{1}" when trying to talk to the Docker daemon: {0}'.format(e, self.get_option('remote_addr')))