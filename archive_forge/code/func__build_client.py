from __future__ import annotations
import traceback
from typing import Any, NoReturn
from ansible.module_utils.basic import AnsibleModule as AnsibleModuleBase, env_fallback
from ansible.module_utils.common.text.converters import to_native
from .client import ClientException, client_check_required_lib, client_get_by_name_or_id
from .vendor.hcloud import APIException, Client, HCloudException
from .vendor.hcloud.actions import ActionException
from .version import version
def _build_client(self) -> None:
    self.client = Client(token=self.module.params['api_token'], api_endpoint=self.module.params['api_endpoint'], application_name='ansible-module', application_version=version)