from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.configuration import ParamName, PATH_PARAMS_FOR_DEFAULT_OBJ
class FtdOperations:
    """
    Utility class for common operation names
    """
    GET_SYSTEM_INFO = 'getSystemInformation'
    GET_MANAGEMENT_IP_LIST = 'getManagementIPList'
    GET_DNS_SETTING_LIST = 'getDeviceDNSSettingsList'
    GET_DNS_SERVER_GROUP = 'getDNSServerGroup'