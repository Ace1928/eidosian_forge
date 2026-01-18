from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
class PodmanPodDiff:

    def __init__(self, module, module_params, info, infra_info, podman_version):
        self.module = module
        self.module_params = module_params
        self.version = podman_version
        self.default_dict = None
        self.info = lower_keys(info)
        self.infra_info = lower_keys(infra_info)
        self.params = self.defaultize()
        self.diff = {'before': {}, 'after': {}}
        self.non_idempotent = {}

    def defaultize(self):
        params_with_defaults = {}
        self.default_dict = PodmanPodDefaults(self.module, self.version).default_dict()
        for p in self.module_params:
            if self.module_params[p] is None and p in self.default_dict:
                params_with_defaults[p] = self.default_dict[p]
            else:
                params_with_defaults[p] = self.module_params[p]
        return params_with_defaults

    def _diff_update_and_compare(self, param_name, before, after):
        if before != after:
            self.diff['before'].update({param_name: before})
            self.diff['after'].update({param_name: after})
            return True
        return False

    def diffparam_add_host(self):
        if not self.infra_info:
            return self._diff_update_and_compare('add_host', '', '')
        before = self.infra_info['hostconfig']['extrahosts'] or []
        after = self.params['add_host']
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('add_host', before, after)

    def diffparam_cgroup_parent(self):
        before = self.info.get('cgroupparent', '') or self.info.get('hostconfig', {}).get('cgroupparent', '')
        after = self.params['cgroup_parent'] or before
        return self._diff_update_and_compare('cgroup_parent', before, after)

    def diffparam_dns(self):
        if not self.infra_info:
            return self._diff_update_and_compare('dns', '', '')
        before = self.infra_info['hostconfig']['dns'] or []
        after = self.params['dns']
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('dns', before, after)

    def diffparam_dns_opt(self):
        if not self.infra_info:
            return self._diff_update_and_compare('dns_opt', '', '')
        before = self.infra_info['hostconfig']['dnsoptions'] or []
        after = self.params['dns_opt']
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('dns_opt', before, after)

    def diffparam_dns_search(self):
        if not self.infra_info:
            return self._diff_update_and_compare('dns_search', '', '')
        before = self.infra_info['hostconfig']['dnssearch'] or []
        after = self.params['dns_search']
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('dns_search', before, after)

    def diffparam_hostname(self):
        if not self.infra_info:
            return self._diff_update_and_compare('hostname', '', '')
        before = self.infra_info['config']['hostname']
        after = self.params['hostname'] or before
        return self._diff_update_and_compare('hostname', before, after)

    def diffparam_infra(self):
        if 'state' in self.info and 'infracontainerid' in self.info['state']:
            before = self.info['state']['infracontainerid'] != ''
        else:
            before = 'infracontainerid' in self.info
        after = self.params['infra']
        return self._diff_update_and_compare('infra', before, after)

    def diffparam_infra_image(self):
        if not self.infra_info:
            return self._diff_update_and_compare('infra_image', '', '')
        before = str(self.infra_info['imagename'])
        after = before
        if self.module_params['infra_image']:
            after = self.params['infra_image']
        before = before.replace(':latest', '')
        after = after.replace(':latest', '')
        before = before.split('/')[-1]
        after = after.split('/')[-1]
        return self._diff_update_and_compare('infra_image', before, after)

    def diffparam_label(self):
        if 'config' in self.info and 'labels' in self.info['config']:
            before = self.info['config'].get('labels') or {}
        else:
            before = self.info['labels'] if 'labels' in self.info else {}
        after = self.params['label']
        if 'podman_systemd_unit' in before:
            after.pop('podman_systemd_unit', None)
            before.pop('podman_systemd_unit', None)
        return self._diff_update_and_compare('label', before, after)

    def diffparam_network(self):
        if not self.infra_info:
            return self._diff_update_and_compare('network', [], [])
        net_mode_before = self.infra_info['hostconfig']['networkmode']
        net_mode_after = ''
        before = list(self.infra_info['networksettings'].get('networks', {}))
        if before == ['podman']:
            before = []
        after = self.params['network'] or []
        if net_mode_before == 'slirp4netns' and 'createcommand' in self.info:
            cr_com = self.info['createcommand']
            if '--network' in cr_com:
                cr_net = cr_com[cr_com.index('--network') + 1].lower()
                if 'slirp4netns:' in cr_net:
                    before = [cr_net]
        if after in [['bridge'], ['host'], ['slirp4netns']]:
            net_mode_after = after[0]
        if net_mode_after and (not before):
            net_mode_after = net_mode_after.replace('bridge', 'default')
            net_mode_after = net_mode_after.replace('slirp4netns', 'default')
            net_mode_before = net_mode_before.replace('bridge', 'default')
            net_mode_before = net_mode_before.replace('slirp4netns', 'default')
            return self._diff_update_and_compare('network', net_mode_before, net_mode_after)
        if not net_mode_after and net_mode_before == 'slirp4netns' and (not after):
            net_mode_after = 'slirp4netns'
            if before == ['slirp4netns']:
                after = ['slirp4netns']
        if not net_mode_after and net_mode_before == 'bridge' and (not after):
            net_mode_after = 'bridge'
            if before == ['bridge']:
                after = ['bridge']
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('network', before, after)

    def diffparam_publish(self):

        def compose(p, h):
            s = ':'.join([str(h['hostport']), p.replace('/tcp', '')]).strip(':')
            if h['hostip']:
                return ':'.join([h['hostip'], s])
            return s
        if not self.infra_info:
            return self._diff_update_and_compare('publish', '', '')
        ports = self.infra_info['hostconfig']['portbindings']
        before = []
        for port, hosts in ports.items():
            if hosts:
                for h in hosts:
                    before.append(compose(port, h))
        after = self.params['publish'] or []
        after = [i.replace('/tcp', '').replace('[', '').replace(']', '') for i in after]
        for ports in after:
            if '-' in ports:
                return self._diff_update_and_compare('publish', '', '')
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('publish', before, after)

    def diffparam_share(self):
        if not self.infra_info:
            return self._diff_update_and_compare('share', '', '')
        if 'sharednamespaces' in self.info:
            before = self.info['sharednamespaces']
        elif 'config' in self.info:
            before = [i.split('shares')[1].lower() for i in self.info['config'] if 'shares' in i]
            before.remove('cgroup')
        else:
            before = []
        if self.params['share'] is not None:
            after = self.params['share'].split(',')
        else:
            after = ['uts', 'ipc', 'net']
            if 'net' not in before:
                after.remove('net')
        if self.params['uidmap'] or self.params['gidmap'] or self.params['userns']:
            after.append('user')
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('share', before, after)

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
            if self.module_params[p] is not None and self.module_params[p] not in [{}, [], '']:
                different = True
        return different