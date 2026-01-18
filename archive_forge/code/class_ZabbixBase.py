from __future__ import absolute_import, division, print_function
from ansible_collections.community.zabbix.plugins.module_utils.api_request import ZabbixApiRequest
class ZabbixBase(object):
    """
    The base class for deriving off module classes
    """

    def __init__(self, module, zbx=None, zapi_wrapper=None):
        self._module = module
        self._zapi = ZabbixApiRequest(module)
        self._zbx_api_version = self._zapi.api_version()