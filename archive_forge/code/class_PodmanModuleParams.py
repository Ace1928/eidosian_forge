from __future__ import (absolute_import, division, print_function)
import json  # noqa: F402
import os  # noqa: F402
import shlex  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import normalize_signal
from ansible_collections.containers.podman.plugins.module_utils.podman.common import ARGUMENTS_OPTS_DICT
class PodmanModuleParams:
    """Creates list of arguments for podman CLI command.

       Arguments:
           action {str} -- action type from 'run', 'stop', 'create', 'delete',
                           'start', 'restart'
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
        if self.action in ['start', 'stop', 'delete', 'restart']:
            return self.start_stop_delete()
        if self.action in ['create', 'run']:
            cmd = [self.action, '--name', self.params['name']]
            all_param_methods = [func for func in dir(self) if callable(getattr(self, func)) and func.startswith('addparam')]
            params_set = (i for i in self.params if self.params[i] is not None)
            for param in params_set:
                func_name = '_'.join(['addparam', param])
                if func_name in all_param_methods:
                    cmd = getattr(self, func_name)(cmd)
            cmd.append(self.params['image'])
            if self.params['command']:
                if isinstance(self.params['command'], list):
                    cmd += self.params['command']
                else:
                    cmd += self.params['command'].split()
            return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]

    def start_stop_delete(self):

        def complete_params(cmd):
            if self.params['attach'] and self.action == 'start':
                cmd.append('--attach')
            if self.params['detach'] is False and self.action == 'start' and ('--attach' not in cmd):
                cmd.append('--attach')
            if self.params['detach_keys'] and self.action == 'start':
                cmd += ['--detach-keys', self.params['detach_keys']]
            if self.params['sig_proxy'] and self.action == 'start':
                cmd.append('--sig-proxy')
            if self.params['stop_time'] and self.action == 'stop':
                cmd += ['--time', self.params['stop_time']]
            if self.params['restart_time'] and self.action == 'restart':
                cmd += ['--time', self.params['restart_time']]
            if self.params['delete_depend'] and self.action == 'delete':
                cmd.append('--depend')
            if self.params['delete_time'] and self.action == 'delete':
                cmd += ['--time', self.params['delete_time']]
            if self.params['delete_volumes'] and self.action == 'delete':
                cmd.append('--volumes')
            if self.params['force_delete'] and self.action == 'delete':
                cmd.append('--force')
            return cmd
        if self.action in ['stop', 'start', 'restart']:
            cmd = complete_params([self.action]) + [self.params['name']]
            return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]
        if self.action == 'delete':
            cmd = complete_params(['rm']) + [self.params['name']]
            return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]

    def check_version(self, param, minv=None, maxv=None):
        if minv and LooseVersion(minv) > LooseVersion(self.podman_version):
            self.module.fail_json(msg='Parameter %s is supported from podman version %s only! Current version is %s' % (param, minv, self.podman_version))
        if maxv and LooseVersion(maxv) < LooseVersion(self.podman_version):
            self.module.fail_json(msg='Parameter %s is supported till podman version %s only! Current version is %s' % (param, minv, self.podman_version))

    def addparam_annotation(self, c):
        for annotate in self.params['annotation'].items():
            c += ['--annotation', '='.join(annotate)]
        return c

    def addparam_attach(self, c):
        for attach in self.params['attach']:
            c += ['--attach=%s' % attach]
        return c

    def addparam_authfile(self, c):
        return c + ['--authfile', self.params['authfile']]

    def addparam_blkio_weight(self, c):
        return c + ['--blkio-weight', self.params['blkio_weight']]

    def addparam_blkio_weight_device(self, c):
        for blkio in self.params['blkio_weight_device'].items():
            c += ['--blkio-weight-device', ':'.join(blkio)]
        return c

    def addparam_cap_add(self, c):
        for cap_add in self.params['cap_add']:
            c += ['--cap-add', cap_add]
        return c

    def addparam_cap_drop(self, c):
        for cap_drop in self.params['cap_drop']:
            c += ['--cap-drop', cap_drop]
        return c

    def addparam_cgroups(self, c):
        self.check_version('--cgroups', minv='1.6.0')
        return c + ['--cgroups=%s' % self.params['cgroups']]

    def addparam_cgroupns(self, c):
        self.check_version('--cgroupns', minv='1.6.2')
        return c + ['--cgroupns=%s' % self.params['cgroupns']]

    def addparam_cgroup_parent(self, c):
        return c + ['--cgroup-parent', self.params['cgroup_parent']]

    def addparam_cidfile(self, c):
        return c + ['--cidfile', self.params['cidfile']]

    def addparam_conmon_pidfile(self, c):
        return c + ['--conmon-pidfile', self.params['conmon_pidfile']]

    def addparam_cpu_period(self, c):
        return c + ['--cpu-period', self.params['cpu_period']]

    def addparam_cpu_quota(self, c):
        return c + ['--cpu-quota', self.params['cpu_quota']]

    def addparam_cpu_rt_period(self, c):
        return c + ['--cpu-rt-period', self.params['cpu_rt_period']]

    def addparam_cpu_rt_runtime(self, c):
        return c + ['--cpu-rt-runtime', self.params['cpu_rt_runtime']]

    def addparam_cpu_shares(self, c):
        return c + ['--cpu-shares', self.params['cpu_shares']]

    def addparam_cpus(self, c):
        return c + ['--cpus', self.params['cpus']]

    def addparam_cpuset_cpus(self, c):
        return c + ['--cpuset-cpus', self.params['cpuset_cpus']]

    def addparam_cpuset_mems(self, c):
        return c + ['--cpuset-mems', self.params['cpuset_mems']]

    def addparam_detach(self, c):
        if self.action == 'create' or self.params['attach']:
            return c
        return c + ['--detach=%s' % self.params['detach']]

    def addparam_detach_keys(self, c):
        return c + ['--detach-keys', self.params['detach_keys']]

    def addparam_device(self, c):
        for dev in self.params['device']:
            c += ['--device', dev]
        return c

    def addparam_device_read_bps(self, c):
        for dev in self.params['device_read_bps']:
            c += ['--device-read-bps', dev]
        return c

    def addparam_device_read_iops(self, c):
        for dev in self.params['device_read_iops']:
            c += ['--device-read-iops', dev]
        return c

    def addparam_device_write_bps(self, c):
        for dev in self.params['device_write_bps']:
            c += ['--device-write-bps', dev]
        return c

    def addparam_device_write_iops(self, c):
        for dev in self.params['device_write_iops']:
            c += ['--device-write-iops', dev]
        return c

    def addparam_dns(self, c):
        return c + ['--dns', ','.join(self.params['dns'])]

    def addparam_dns_option(self, c):
        return c + ['--dns-option', self.params['dns_option']]

    def addparam_dns_search(self, c):
        return c + ['--dns-search', self.params['dns_search']]

    def addparam_entrypoint(self, c):
        return c + ['--entrypoint', self.params['entrypoint']]

    def addparam_env(self, c):
        for env_value in self.params['env'].items():
            c += ['--env', b'='.join([to_bytes(k, errors='surrogate_or_strict') for k in env_value])]
        return c

    def addparam_env_file(self, c):
        for env_file in self.params['env_file']:
            c += ['--env-file', env_file]
        return c

    def addparam_env_host(self, c):
        self.check_version('--env-host', minv='1.5.0')
        return c + ['--env-host=%s' % self.params['env_host']]

    def addparam_etc_hosts(self, c):
        for host_ip in self.params['etc_hosts'].items():
            c += ['--add-host', ':'.join(host_ip)]
        return c

    def addparam_expose(self, c):
        for exp in self.params['expose']:
            c += ['--expose', exp]
        return c

    def addparam_gidmap(self, c):
        for gidmap in self.params['gidmap']:
            c += ['--gidmap', gidmap]
        return c

    def addparam_group_add(self, c):
        for g in self.params['group_add']:
            c += ['--group-add', g]
        return c

    def addparam_healthcheck(self, c):
        return c + ['--healthcheck-command', self.params['healthcheck']]

    def addparam_healthcheck_interval(self, c):
        return c + ['--healthcheck-interval', self.params['healthcheck_interval']]

    def addparam_healthcheck_retries(self, c):
        return c + ['--healthcheck-retries', self.params['healthcheck_retries']]

    def addparam_healthcheck_start_period(self, c):
        return c + ['--healthcheck-start-period', self.params['healthcheck_start_period']]

    def addparam_healthcheck_timeout(self, c):
        return c + ['--healthcheck-timeout', self.params['healthcheck_timeout']]

    def addparam_healthcheck_failure_action(self, c):
        return c + ['--health-on-failure', self.params['healthcheck_failure_action']]

    def addparam_hooks_dir(self, c):
        for hook_dir in self.params['hooks_dir']:
            c += ['--hooks-dir=%s' % hook_dir]
        return c

    def addparam_hostname(self, c):
        return c + ['--hostname', self.params['hostname']]

    def addparam_http_proxy(self, c):
        return c + ['--http-proxy=%s' % self.params['http_proxy']]

    def addparam_image_volume(self, c):
        return c + ['--image-volume', self.params['image_volume']]

    def addparam_init(self, c):
        if self.params['init']:
            c += ['--init']
        return c

    def addparam_init_path(self, c):
        return c + ['--init-path', self.params['init_path']]

    def addparam_interactive(self, c):
        return c + ['--interactive=%s' % self.params['interactive']]

    def addparam_ip(self, c):
        return c + ['--ip', self.params['ip']]

    def addparam_ipc(self, c):
        return c + ['--ipc', self.params['ipc']]

    def addparam_kernel_memory(self, c):
        return c + ['--kernel-memory', self.params['kernel_memory']]

    def addparam_label(self, c):
        for label in self.params['label'].items():
            c += ['--label', b'='.join([to_bytes(la, errors='surrogate_or_strict') for la in label])]
        return c

    def addparam_label_file(self, c):
        return c + ['--label-file', self.params['label_file']]

    def addparam_log_driver(self, c):
        return c + ['--log-driver', self.params['log_driver']]

    def addparam_log_opt(self, c):
        for k, v in self.params['log_opt'].items():
            if v is not None:
                c += ['--log-opt', b'='.join([to_bytes(k.replace('max_size', 'max-size'), errors='surrogate_or_strict'), to_bytes(v, errors='surrogate_or_strict')])]
        return c

    def addparam_log_level(self, c):
        return c + ['--log-level', self.params['log_level']]

    def addparam_mac_address(self, c):
        return c + ['--mac-address', self.params['mac_address']]

    def addparam_memory(self, c):
        return c + ['--memory', self.params['memory']]

    def addparam_memory_reservation(self, c):
        return c + ['--memory-reservation', self.params['memory_reservation']]

    def addparam_memory_swap(self, c):
        return c + ['--memory-swap', self.params['memory_swap']]

    def addparam_memory_swappiness(self, c):
        return c + ['--memory-swappiness', self.params['memory_swappiness']]

    def addparam_mount(self, c):
        for mnt in self.params['mount']:
            if mnt:
                c += ['--mount', mnt]
        return c

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
        return c + ['--no-hosts=%s' % self.params['no_hosts']]

    def addparam_oom_kill_disable(self, c):
        return c + ['--oom-kill-disable=%s' % self.params['oom_kill_disable']]

    def addparam_oom_score_adj(self, c):
        return c + ['--oom-score-adj', self.params['oom_score_adj']]

    def addparam_pid(self, c):
        return c + ['--pid', self.params['pid']]

    def addparam_pids_limit(self, c):
        return c + ['--pids-limit', self.params['pids_limit']]

    def addparam_pod(self, c):
        return c + ['--pod', self.params['pod']]

    def addparam_privileged(self, c):
        return c + ['--privileged=%s' % self.params['privileged']]

    def addparam_publish(self, c):
        for pub in self.params['publish']:
            c += ['--publish', pub]
        return c

    def addparam_publish_all(self, c):
        return c + ['--publish-all=%s' % self.params['publish_all']]

    def addparam_read_only(self, c):
        return c + ['--read-only=%s' % self.params['read_only']]

    def addparam_read_only_tmpfs(self, c):
        return c + ['--read-only-tmpfs=%s' % self.params['read_only_tmpfs']]

    def addparam_requires(self, c):
        return c + ['--requires', ','.join(self.params['requires'])]

    def addparam_restart_policy(self, c):
        return c + ['--restart=%s' % self.params['restart_policy']]

    def addparam_rm(self, c):
        if self.params['rm']:
            c += ['--rm']
        return c

    def addparam_rootfs(self, c):
        return c + ['--rootfs=%s' % self.params['rootfs']]

    def addparam_sdnotify(self, c):
        return c + ['--sdnotify=%s' % self.params['sdnotify']]

    def addparam_secrets(self, c):
        for secret in self.params['secrets']:
            c += ['--secret', secret]
        return c

    def addparam_security_opt(self, c):
        for secopt in self.params['security_opt']:
            c += ['--security-opt', secopt]
        return c

    def addparam_shm_size(self, c):
        return c + ['--shm-size', self.params['shm_size']]

    def addparam_sig_proxy(self, c):
        return c + ['--sig-proxy=%s' % self.params['sig_proxy']]

    def addparam_stop_signal(self, c):
        return c + ['--stop-signal', self.params['stop_signal']]

    def addparam_stop_timeout(self, c):
        return c + ['--stop-timeout', self.params['stop_timeout']]

    def addparam_subgidname(self, c):
        return c + ['--subgidname', self.params['subgidname']]

    def addparam_subuidname(self, c):
        return c + ['--subuidname', self.params['subuidname']]

    def addparam_sysctl(self, c):
        for sysctl in self.params['sysctl'].items():
            c += ['--sysctl', b'='.join([to_bytes(k, errors='surrogate_or_strict') for k in sysctl])]
        return c

    def addparam_systemd(self, c):
        return c + ['--systemd=%s' % str(self.params['systemd']).lower()]

    def addparam_tmpfs(self, c):
        for tmpfs in self.params['tmpfs'].items():
            c += ['--tmpfs', ':'.join(tmpfs)]
        return c

    def addparam_timezone(self, c):
        return c + ['--tz=%s' % self.params['timezone']]

    def addparam_tty(self, c):
        return c + ['--tty=%s' % self.params['tty']]

    def addparam_uidmap(self, c):
        for uidmap in self.params['uidmap']:
            c += ['--uidmap', uidmap]
        return c

    def addparam_ulimit(self, c):
        for u in self.params['ulimit']:
            c += ['--ulimit', u]
        return c

    def addparam_user(self, c):
        return c + ['--user', self.params['user']]

    def addparam_userns(self, c):
        return c + ['--userns', self.params['userns']]

    def addparam_uts(self, c):
        return c + ['--uts', self.params['uts']]

    def addparam_volume(self, c):
        for vol in self.params['volume']:
            if vol:
                c += ['--volume', vol]
        return c

    def addparam_volumes_from(self, c):
        for vol in self.params['volumes_from']:
            c += ['--volumes-from', vol]
        return c

    def addparam_workdir(self, c):
        return c + ['--workdir', self.params['workdir']]

    def addparam_cmd_args(self, c):
        return c + self.params['cmd_args']