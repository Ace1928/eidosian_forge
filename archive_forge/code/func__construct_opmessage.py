from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_opmessage(self, operation):
    """Construct operation message.

        Args:
            operation: operation to construct the message

        Returns:
            dict: constructed operation message
        """
    try:
        return {'default_msg': '0' if operation.get('op_message') is not None or operation.get('subject') is not None else '1', 'mediatypeid': self._zapi_wrapper.get_mediatype_by_mediatype_name(operation.get('media_type')) if operation.get('media_type') is not None else '0', 'message': operation.get('op_message'), 'subject': operation.get('subject')}
    except Exception as e:
        self._module.fail_json(msg='Failed to construct operation message. The error was: %s' % e)