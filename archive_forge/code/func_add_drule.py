from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def add_drule(self, **kwargs):
    """Add discovery rule
        Args:
            **kwargs: Arbitrary keyword parameters
        Returns:
            drule: created discovery rule
        """
    try:
        if self._module.check_mode:
            self._module.exit_json(msg='Discovery rule would be added if check mode was not specified', changed=True)
        parameters = self._construct_parameters(**kwargs)
        drule_list = self._zapi.drule.create(parameters)
        return drule_list['druleids'][0]
    except Exception as e:
        self._module.fail_json(msg='Failed to create discovery rule %s: %s' % (kwargs['name'], e))