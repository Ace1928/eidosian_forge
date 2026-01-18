from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
class PodmanPodModuleParams:
    """Creates list of arguments for podman CLI command.

       Arguments:
           action {str} -- action type from 'run', 'stop', 'create', 'delete',
                           'start'
           params {dict} -- dictionary of module parameters

       """

    def __init__(self, action, params, podman_version, module):
        self.params = params
        self.action = action
        self.podman_version = podman_version
        self.module = module

    def construct_command_from_params(self):
        """Create a podman command from given module parameters.

        Returns:
           list -- list of byte strings for Popen command
        """
        if self.action in ['start', 'restart', 'stop', 'delete', 'pause', 'unpause', 'kill']:
            return self._simple_action()
        if self.action in ['create']:
            return self._create_action()
        self.module.fail_json(msg='Unknown action %s' % self.action)

    def _simple_action(self):
        if self.action in ['start', 'restart', 'stop', 'pause', 'unpause', 'kill']:
            cmd = [self.action, self.params['name']]
            return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]
        if self.action == 'delete':
            cmd = ['rm', '-f', self.params['name']]
            return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]
        self.module.fail_json(msg='Unknown action %s' % self.action)

    def _create_action(self):
        cmd = [self.action]
        all_param_methods = [func for func in dir(self) if callable(getattr(self, func)) and func.startswith('addparam')]
        params_set = (i for i in self.params if self.params[i] is not None)
        for param in params_set:
            func_name = '_'.join(['addparam', param])
            if func_name in all_param_methods:
                cmd = getattr(self, func_name)(cmd)
        return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]

    def check_version(self, param, minv=None, maxv=None):
        if minv and LooseVersion(minv) > LooseVersion(self.podman_version):
            self.module.fail_json(msg='Parameter %s is supported from podman version %s only! Current version is %s' % (param, minv, self.podman_version))
        if maxv and LooseVersion(maxv) < LooseVersion(self.podman_version):
            self.module.fail_json(msg='Parameter %s is supported till podman version %s only! Current version is %s' % (param, minv, self.podman_version))

    def addparam_add_host(self, c):
        for g in self.params['add_host']:
            c += ['--add-host', g]
        return c

    def addparam_blkio_weight(self, c):
        self.check_version('--blkio-weight', minv='4.3.0')
        return c + ['--blkio-weight', self.params['blkio_weight']]

    def addparam_blkio_weight_device(self, c):
        self.check_version('--blkio-weight-device', minv='4.3.0')
        for dev in self.params['blkio_weight_device']:
            c += ['--blkio-weight-device', dev]
        return c

    def addparam_cgroup_parent(self, c):
        return c + ['--cgroup-parent', self.params['cgroup_parent']]

    def addparam_cpus(self, c):
        self.check_version('--cpus', minv='4.2.0')
        return c + ['--cpus', self.params['cpus']]

    def addparam_cpuset_cpus(self, c):
        self.check_version('--cpus', minv='4.2.0')
        return c + ['--cpuset-cpus', self.params['cpuset_cpus']]

    def addparam_cpuset_mems(self, c):
        self.check_version('--cpuset-mems', minv='4.3.0')
        return c + ['--cpuset-mems', self.params['cpuset_mems']]

    def addparam_cpu_shares(self, c):
        self.check_version('--cpu-shares', minv='4.3.0')
        return c + ['--cpu-shares', self.params['cpu_shares']]

    def addparam_device(self, c):
        for dev in self.params['device']:
            c += ['--device', dev]
        return c

    def addparam_device_read_bps(self, c):
        self.check_version('--device-read-bps', minv='4.3.0')
        for dev in self.params['device_read_bps']:
            c += ['--device-read-bps', dev]
        return c

    def addparam_device_write_bps(self, c):
        self.check_version('--device-write-bps', minv='4.3.0')
        for dev in self.params['device_write_bps']:
            c += ['--device-write-bps', dev]
        return c

    def addparam_dns(self, c):
        for g in self.params['dns']:
            c += ['--dns', g]
        return c

    def addparam_dns_opt(self, c):
        for g in self.params['dns_opt']:
            c += ['--dns-opt', g]
        return c

    def addparam_dns_search(self, c):
        for g in self.params['dns_search']:
            c += ['--dns-search', g]
        return c

    def addparam_gidmap(self, c):
        for gidmap in self.params['gidmap']:
            c += ['--gidmap', gidmap]
        return c

    def addparam_hostname(self, c):
        return c + ['--hostname', self.params['hostname']]

    def addparam_infra(self, c):
        return c + [b'='.join([b'--infra', to_bytes(self.params['infra'], errors='surrogate_or_strict')])]

    def addparam_infra_conmon_pidfile(self, c):
        return c + ['--infra-conmon-pidfile', self.params['infra_conmon_pidfile']]

    def addparam_infra_command(self, c):
        return c + ['--infra-command', self.params['infra_command']]

    def addparam_infra_image(self, c):
        return c + ['--infra-image', self.params['infra_image']]

    def addparam_infra_name(self, c):
        return c + ['--infra-name', self.params['infra_name']]

    def addparam_ip(self, c):
        return c + ['--ip', self.params['ip']]

    def addparam_label(self, c):
        for label in self.params['label'].items():
            c += ['--label', b'='.join([to_bytes(i, errors='surrogate_or_strict') for i in label])]
        return c

    def addparam_label_file(self, c):
        return c + ['--label-file', self.params['label_file']]

    def addparam_mac_address(self, c):
        return c + ['--mac-address', self.params['mac_address']]

    def addparam_memory(self, c):
        self.check_version('--memory', minv='4.2.0')
        return c + ['--memory', self.params['memory']]

    def addparam_memory_swap(self, c):
        self.check_version('--memory-swap', minv='4.3.0')
        return c + ['--memory-swap', self.params['memory_swap']]

    def addparam_name(self, c):
        return c + ['--name', self.params['name']]

    def addparam_network(self, c):
        if LooseVersion(self.podman_version) >= LooseVersion('4.0.0'):
            for net in self.params['network']:
                c += ['--network', net]
            return c
        return c + ['--network', ','.join(self.params['network'])]

    def addparam_network_aliases(self, c):
        for alias in self.params['network_aliases']:
            c += ['--network-alias', alias]
        return c

    def addparam_no_hosts(self, c):
        return c + ['='.join('--no-hosts', self.params['no_hosts'])]

    def addparam_pid(self, c):
        return c + ['--pid', self.params['pid']]

    def addparam_pod_id_file(self, c):
        return c + ['--pod-id-file', self.params['pod_id_file']]

    def addparam_publish(self, c):
        for g in self.params['publish']:
            c += ['--publish', g]
        return c

    def addparam_share(self, c):
        return c + ['--share', self.params['share']]

    def addparam_subgidname(self, c):
        return c + ['--subgidname', self.params['subgidname']]

    def addparam_subuidname(self, c):
        return c + ['--subuidname', self.params['subuidname']]

    def addparam_uidmap(self, c):
        for uidmap in self.params['uidmap']:
            c += ['--uidmap', uidmap]
        return c

    def addparam_userns(self, c):
        return c + ['--userns', self.params['userns']]

    def addparam_volume(self, c):
        for vol in self.params['volume']:
            if vol:
                c += ['--volume', vol]
        return c