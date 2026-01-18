from __future__ import annotations
from contextlib import contextmanager
from ansible.module_utils.basic import missing_required_lib
from .vendor.hcloud import APIException, Client as ClientBase
def _client_resource_not_found(resource: str, param: str | int):
    return ClientException(f'resource ({resource.rstrip('s')}) does not exist: {param}')