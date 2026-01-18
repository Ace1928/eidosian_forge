from __future__ import annotations
import os
import sys
from ipaddress import IPv6Network
from ansible.errors import AnsibleError
from ansible.inventory.manager import InventoryData
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, Constructable
from ansible.utils.display import Display
from ..module_utils.client import (
from ..module_utils.vendor.hcloud import APIException
from ..module_utils.vendor.hcloud.networks import Network
from ..module_utils.vendor.hcloud.servers import Server
from ..module_utils.version import version
def _get_cached_result(self, path, cache) -> tuple[list[InventoryServer], bool]:
    if not cache:
        return ([], False)
    if not self.get_option('cache'):
        return ([], False)
    cache_key = self.get_cache_key(path)
    try:
        cached_result = self._cache[cache_key]
    except KeyError:
        return ([], False)
    return (cached_result, True)