from __future__ import annotations
import traceback
from typing import Any, NoReturn
from ansible.module_utils.basic import AnsibleModule as AnsibleModuleBase, env_fallback
from ansible.module_utils.common.text.converters import to_native
from .client import ClientException, client_check_required_lib, client_get_by_name_or_id
from .vendor.hcloud import APIException, Client, HCloudException
from .vendor.hcloud.actions import ActionException
from .version import version
def _client_get_by_name_or_id(self, resource: str, param: str | int):
    """
        Get a resource by name, and if not found by its ID.

        :param resource: Name of the resource client that implements both `get_by_name` and `get_by_id` methods
        :param param: Name or ID of the resource to query
        """
    try:
        return client_get_by_name_or_id(self.client, resource, param)
    except ClientException as exception:
        self.module.fail_json(msg=to_native(exception))