from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
class PodmanNetworkDiff:

    def __init__(self, module, info, podman_version):
        self.module = module
        self.version = podman_version
        self.default_dict = None
        self.info = lower_keys(info)
        self.params = self.defaultize()
        self.diff = {'before': {}, 'after': {}}
        self.non_idempotent = {}

    def defaultize(self):
        params_with_defaults = {}
        self.default_dict = PodmanNetworkDefaults(self.module, self.version).default_dict()
        for p in self.module.params:
            if self.module.params[p] is None and p in self.default_dict:
                params_with_defaults[p] = self.default_dict[p]
            else:
                params_with_defaults[p] = self.module.params[p]
        return params_with_defaults

    def _diff_update_and_compare(self, param_name, before, after):
        if before != after:
            self.diff['before'].update({param_name: before})
            self.diff['after'].update({param_name: after})
            return True
        return False

    def diffparam_disable_dns(self):
        if LooseVersion(self.version) >= LooseVersion('4.0.0'):
            before = not self.info.get('dns_enabled', True)
            after = self.params['disable_dns']
            if self.params['disable_dns'] is None:
                after = before
            return self._diff_update_and_compare('disable_dns', before, after)
        before = after = self.params['disable_dns']
        return self._diff_update_and_compare('disable_dns', before, after)

    def diffparam_driver(self):
        before = after = 'bridge'
        return self._diff_update_and_compare('driver', before, after)

    def diffparam_ipv6(self):
        if LooseVersion(self.version) >= LooseVersion('4.0.0'):
            before = self.info.get('ipv6_enabled', False)
            after = self.params['ipv6']
            return self._diff_update_and_compare('ipv6', before, after)
        before = after = ''
        return self._diff_update_and_compare('ipv6', before, after)

    def diffparam_gateway(self):
        if LooseVersion(self.version) >= LooseVersion('4.0.0'):
            return self._diff_update_and_compare('gateway', '', '')
        try:
            before = self.info['plugins'][0]['ipam']['ranges'][0][0]['gateway']
        except (IndexError, KeyError):
            before = ''
        after = before
        if self.params['gateway'] is not None:
            after = self.params['gateway']
        return self._diff_update_and_compare('gateway', before, after)

    def diffparam_internal(self):
        if LooseVersion(self.version) >= LooseVersion('4.0.0'):
            before = self.info.get('internal', False)
            after = self.params['internal']
            return self._diff_update_and_compare('internal', before, after)
        try:
            before = not self.info['plugins'][0]['isgateway']
        except (IndexError, KeyError):
            before = False
        after = self.params['internal']
        return self._diff_update_and_compare('internal', before, after)

    def diffparam_ip_range(self):
        before = after = ''
        return self._diff_update_and_compare('ip_range', before, after)

    def diffparam_subnet(self):
        if LooseVersion(self.version) >= LooseVersion('4.0.0'):
            return self._diff_update_and_compare('subnet', '', '')
        try:
            before = self.info['plugins'][0]['ipam']['ranges'][0][0]['subnet']
        except (IndexError, KeyError):
            before = ''
        after = before
        if self.params['subnet'] is not None:
            after = self.params['subnet']
            if HAS_IP_ADDRESS_MODULE:
                after = ipaddress.ip_network(after).compressed
        return self._diff_update_and_compare('subnet', before, after)

    def diffparam_macvlan(self):
        before = after = ''
        return self._diff_update_and_compare('macvlan', before, after)

    def diffparam_opt(self):
        if LooseVersion(self.version) >= LooseVersion('4.0.0'):
            vlan_before = self.info.get('options', {}).get('vlan')
        else:
            try:
                vlan_before = self.info['plugins'][0].get('vlan')
            except (IndexError, KeyError):
                vlan_before = None
        vlan_after = self.params['opt'].get('vlan') if self.params['opt'] else None
        if vlan_before or vlan_after:
            before, after = ({'vlan': str(vlan_before)}, {'vlan': str(vlan_after)})
        else:
            before, after = ({}, {})
        if LooseVersion(self.version) >= LooseVersion('4.0.0'):
            mtu_before = self.info.get('options', {}).get('mtu')
        else:
            try:
                mtu_before = self.info['plugins'][0].get('mtu')
            except (IndexError, KeyError):
                mtu_before = None
        mtu_after = self.params['opt'].get('mtu') if self.params['opt'] else None
        if mtu_before or mtu_after:
            before.update({'mtu': str(mtu_before)})
            after.update({'mtu': str(mtu_after)})
        return self._diff_update_and_compare('opt', before, after)

    def is_different(self):
        diff_func_list = [func for func in dir(self) if callable(getattr(self, func)) and func.startswith('diffparam')]
        fail_fast = not bool(self.module._diff)
        different = False
        for func_name in diff_func_list:
            dff_func = getattr(self, func_name)
            if dff_func():
                if fail_fast:
                    return True
                different = True
        for p in self.non_idempotent:
            if self.module.params[p] is not None and self.module.params[p] not in [{}, [], '']:
                different = True
        return different