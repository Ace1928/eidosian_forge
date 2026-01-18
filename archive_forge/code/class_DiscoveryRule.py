from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class DiscoveryRule(ZabbixBase):

    def check_if_drule_exists(self, name):
        """Check if discovery rule exists.
        Args:
            name: Name of the discovery rule.
        Returns:
            The return value. True for success, False otherwise.
        """
        try:
            _drule = self._zapi.drule.get({'output': 'extend', 'selectDChecks': 'extend', 'filter': {'name': [name]}})
            if len(_drule) > 0:
                return _drule
        except Exception as e:
            self._module.fail_json(msg="Failed to check if discovery rule '%s' exists: %s" % (name, e))

    def get_drule_by_drule_name(self, name):
        """Get discovery rule by discovery rule name
        Args:
            name: discovery rule name.
        Returns:
            discovery rule matching discovery rule name
        """
        try:
            drule_list = self._zapi.drule.get({'output': 'extend', 'selectDChecks': 'extend', 'filter': {'name': [name]}})
            if len(drule_list) < 1:
                self._module.fail_json(msg='Discovery rule not found: %s' % name)
            else:
                return drule_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get discovery rule '%s': %s" % (name, e))

    def get_proxy_by_proxy_name(self, proxy_name):
        """Get proxy by proxy name
        Args:
            proxy_name: proxy name.
        Returns:
            proxy matching proxy name
        """
        try:
            proxy_list = self._zapi.proxy.get({'output': 'extend', 'selectInterface': 'extend', 'filter': {'host': [proxy_name]}})
            if len(proxy_list) < 1:
                self._module.fail_json(msg='Proxy not found: %s' % proxy_name)
            else:
                return proxy_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get proxy '%s': %s" % (proxy_name, e))

    def _construct_parameters(self, **kwargs):
        """Construct parameters.
        Args:
            **kwargs: Arbitrary keyword parameters.
        Returns:
            dict: dictionary of specified parameters
        """
        _params = {'name': kwargs['name'], 'iprange': ','.join(kwargs['iprange']), 'delay': kwargs['delay'], 'status': zabbix_utils.helper_to_numeric_value(['enabled', 'disabled'], kwargs['status']), 'dchecks': kwargs['dchecks']}
        if kwargs['proxy']:
            _params['proxy_hostid'] = self.get_proxy_by_proxy_name(kwargs['proxy'])['proxyid']
        return _params

    def check_difference(self, **kwargs):
        """Check difference between discovery rule and user specified parameters.
        Args:
            **kwargs: Arbitrary keyword parameters.
        Returns:
            dict: dictionary of differences
        """
        existing_drule = zabbix_utils.helper_convert_unicode_to_str(self.check_if_drule_exists(kwargs['name'])[0])
        parameters = zabbix_utils.helper_convert_unicode_to_str(self._construct_parameters(**kwargs))
        change_parameters = {}
        if LooseVersion(self._zbx_api_version) < LooseVersion('6.4'):
            if existing_drule['nextcheck']:
                existing_drule.pop('nextcheck')
        _diff = zabbix_utils.helper_cleanup_data(zabbix_utils.helper_compare_dictionaries(parameters, existing_drule, change_parameters))
        return _diff

    def update_drule(self, **kwargs):
        """Update discovery rule.
        Args:
            **kwargs: Arbitrary keyword parameters.
        Returns:
            drule: updated discovery rule
        """
        try:
            if self._module.check_mode:
                self._module.exit_json(msg='Discovery rule would be updated if check mode was not specified: ID %s' % kwargs['drule_id'], changed=True)
            kwargs['druleid'] = kwargs.pop('drule_id')
            return self._zapi.drule.update(kwargs)
        except Exception as e:
            self._module.fail_json(msg="Failed to update discovery rule ID '%s': %s" % (kwargs['drule_id'], e))

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

    def delete_drule(self, drule_id):
        """Delete discovery rule.
        Args:
            drule_id: Discovery rule id
        Returns:
            drule: deleted discovery rule
        """
        try:
            if self._module.check_mode:
                self._module.exit_json(changed=True, msg='Discovery rule would be deleted if check mode was not specified')
            return self._zapi.drule.delete([drule_id])
        except Exception as e:
            self._module.fail_json(msg="Failed to delete discovery rule '%s': %s" % (drule_id, e))