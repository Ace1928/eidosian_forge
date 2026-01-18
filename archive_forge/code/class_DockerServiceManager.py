from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
class DockerServiceManager(object):

    def __init__(self, client):
        self.client = client
        self.retries = 2
        self.diff_tracker = None

    def get_service(self, name):
        try:
            raw_data = self.client.inspect_service(name)
        except NotFound:
            return None
        ds = DockerService(self.client.docker_api_version, self.client.docker_py_version)
        task_template_data = raw_data['Spec']['TaskTemplate']
        ds.image = task_template_data['ContainerSpec']['Image']
        ds.user = task_template_data['ContainerSpec'].get('User')
        ds.env = task_template_data['ContainerSpec'].get('Env')
        ds.command = task_template_data['ContainerSpec'].get('Command')
        ds.args = task_template_data['ContainerSpec'].get('Args')
        ds.groups = task_template_data['ContainerSpec'].get('Groups')
        ds.stop_grace_period = task_template_data['ContainerSpec'].get('StopGracePeriod')
        ds.stop_signal = task_template_data['ContainerSpec'].get('StopSignal')
        ds.working_dir = task_template_data['ContainerSpec'].get('Dir')
        ds.read_only = task_template_data['ContainerSpec'].get('ReadOnly')
        ds.cap_add = task_template_data['ContainerSpec'].get('CapabilityAdd')
        ds.cap_drop = task_template_data['ContainerSpec'].get('CapabilityDrop')
        healthcheck_data = task_template_data['ContainerSpec'].get('Healthcheck')
        if healthcheck_data:
            options = {'Test': 'test', 'Interval': 'interval', 'Timeout': 'timeout', 'StartPeriod': 'start_period', 'Retries': 'retries'}
            healthcheck = dict(((options[key], value) for key, value in healthcheck_data.items() if value is not None and key in options))
            ds.healthcheck = healthcheck
        update_config_data = raw_data['Spec'].get('UpdateConfig')
        if update_config_data:
            ds.update_delay = update_config_data.get('Delay')
            ds.update_parallelism = update_config_data.get('Parallelism')
            ds.update_failure_action = update_config_data.get('FailureAction')
            ds.update_monitor = update_config_data.get('Monitor')
            ds.update_max_failure_ratio = update_config_data.get('MaxFailureRatio')
            ds.update_order = update_config_data.get('Order')
        rollback_config_data = raw_data['Spec'].get('RollbackConfig')
        if rollback_config_data:
            ds.rollback_config = {'parallelism': rollback_config_data.get('Parallelism'), 'delay': rollback_config_data.get('Delay'), 'failure_action': rollback_config_data.get('FailureAction'), 'monitor': rollback_config_data.get('Monitor'), 'max_failure_ratio': rollback_config_data.get('MaxFailureRatio'), 'order': rollback_config_data.get('Order')}
        dns_config = task_template_data['ContainerSpec'].get('DNSConfig')
        if dns_config:
            ds.dns = dns_config.get('Nameservers')
            ds.dns_search = dns_config.get('Search')
            ds.dns_options = dns_config.get('Options')
        ds.hostname = task_template_data['ContainerSpec'].get('Hostname')
        hosts = task_template_data['ContainerSpec'].get('Hosts')
        if hosts:
            hosts = [list(reversed(host.split(':', 1))) if ':' in host else host.split(' ', 1) for host in hosts]
            ds.hosts = dict(((hostname, ip) for ip, hostname in hosts))
        ds.tty = task_template_data['ContainerSpec'].get('TTY')
        placement = task_template_data.get('Placement')
        if placement:
            ds.constraints = placement.get('Constraints')
            ds.replicas_max_per_node = placement.get('MaxReplicas')
            placement_preferences = []
            for preference in placement.get('Preferences', []):
                placement_preferences.append(dict(((key.lower(), value['SpreadDescriptor']) for key, value in preference.items())))
            ds.placement_preferences = placement_preferences or None
        restart_policy_data = task_template_data.get('RestartPolicy')
        if restart_policy_data:
            ds.restart_policy = restart_policy_data.get('Condition')
            ds.restart_policy_delay = restart_policy_data.get('Delay')
            ds.restart_policy_attempts = restart_policy_data.get('MaxAttempts')
            ds.restart_policy_window = restart_policy_data.get('Window')
        raw_data_endpoint_spec = raw_data['Spec'].get('EndpointSpec')
        if raw_data_endpoint_spec:
            ds.endpoint_mode = raw_data_endpoint_spec.get('Mode')
            raw_data_ports = raw_data_endpoint_spec.get('Ports')
            if raw_data_ports:
                ds.publish = []
                for port in raw_data_ports:
                    ds.publish.append({'protocol': port['Protocol'], 'mode': port.get('PublishMode', None), 'published_port': port.get('PublishedPort', None), 'target_port': int(port['TargetPort'])})
        raw_data_limits = task_template_data.get('Resources', {}).get('Limits')
        if raw_data_limits:
            raw_cpu_limits = raw_data_limits.get('NanoCPUs')
            if raw_cpu_limits:
                ds.limit_cpu = float(raw_cpu_limits) / 1000000000
            raw_memory_limits = raw_data_limits.get('MemoryBytes')
            if raw_memory_limits:
                ds.limit_memory = int(raw_memory_limits)
        raw_data_reservations = task_template_data.get('Resources', {}).get('Reservations')
        if raw_data_reservations:
            raw_cpu_reservations = raw_data_reservations.get('NanoCPUs')
            if raw_cpu_reservations:
                ds.reserve_cpu = float(raw_cpu_reservations) / 1000000000
            raw_memory_reservations = raw_data_reservations.get('MemoryBytes')
            if raw_memory_reservations:
                ds.reserve_memory = int(raw_memory_reservations)
        ds.labels = raw_data['Spec'].get('Labels')
        ds.log_driver = task_template_data.get('LogDriver', {}).get('Name')
        ds.log_driver_options = task_template_data.get('LogDriver', {}).get('Options')
        ds.container_labels = task_template_data['ContainerSpec'].get('Labels')
        mode = raw_data['Spec']['Mode']
        if 'Replicated' in mode.keys():
            ds.mode = to_text('replicated', encoding='utf-8')
            ds.replicas = mode['Replicated']['Replicas']
        elif 'Global' in mode.keys():
            ds.mode = 'global'
        else:
            raise Exception('Unknown service mode: %s' % mode)
        raw_data_mounts = task_template_data['ContainerSpec'].get('Mounts')
        if raw_data_mounts:
            ds.mounts = []
            for mount_data in raw_data_mounts:
                bind_options = mount_data.get('BindOptions', {})
                volume_options = mount_data.get('VolumeOptions', {})
                tmpfs_options = mount_data.get('TmpfsOptions', {})
                driver_config = volume_options.get('DriverConfig', {})
                driver_config = dict(((key.lower(), value) for key, value in driver_config.items())) or None
                ds.mounts.append({'source': mount_data.get('Source', ''), 'type': mount_data['Type'], 'target': mount_data['Target'], 'readonly': mount_data.get('ReadOnly'), 'propagation': bind_options.get('Propagation'), 'no_copy': volume_options.get('NoCopy'), 'labels': volume_options.get('Labels'), 'driver_config': driver_config, 'tmpfs_mode': tmpfs_options.get('Mode'), 'tmpfs_size': tmpfs_options.get('SizeBytes')})
        raw_data_configs = task_template_data['ContainerSpec'].get('Configs')
        if raw_data_configs:
            ds.configs = []
            for config_data in raw_data_configs:
                ds.configs.append({'config_id': config_data['ConfigID'], 'config_name': config_data['ConfigName'], 'filename': config_data['File'].get('Name'), 'uid': config_data['File'].get('UID'), 'gid': config_data['File'].get('GID'), 'mode': config_data['File'].get('Mode')})
        raw_data_secrets = task_template_data['ContainerSpec'].get('Secrets')
        if raw_data_secrets:
            ds.secrets = []
            for secret_data in raw_data_secrets:
                ds.secrets.append({'secret_id': secret_data['SecretID'], 'secret_name': secret_data['SecretName'], 'filename': secret_data['File'].get('Name'), 'uid': secret_data['File'].get('UID'), 'gid': secret_data['File'].get('GID'), 'mode': secret_data['File'].get('Mode')})
        raw_networks_data = task_template_data.get('Networks', raw_data['Spec'].get('Networks'))
        if raw_networks_data:
            ds.networks = []
            for network_data in raw_networks_data:
                network = {'id': network_data['Target']}
                if 'Aliases' in network_data:
                    network['aliases'] = network_data['Aliases']
                if 'DriverOpts' in network_data:
                    network['options'] = network_data['DriverOpts']
                ds.networks.append(network)
        ds.service_version = raw_data['Version']['Index']
        ds.service_id = raw_data['ID']
        ds.init = task_template_data['ContainerSpec'].get('Init', False)
        return ds

    def update_service(self, name, old_service, new_service):
        service_data = new_service.build_docker_service()
        result = self.client.update_service(old_service.service_id, old_service.service_version, name=name, **service_data)
        self.client.report_warnings(result, ['Warning'])

    def create_service(self, name, service):
        service_data = service.build_docker_service()
        result = self.client.create_service(name=name, **service_data)
        self.client.report_warnings(result, ['Warning'])

    def remove_service(self, name):
        self.client.remove_service(name)

    def get_image_digest(self, name, resolve=False):
        if not name or not resolve:
            return name
        repo, tag = parse_repository_tag(name)
        if not tag:
            tag = 'latest'
        name = repo + ':' + tag
        distribution_data = self.client.inspect_distribution(name)
        digest = distribution_data['Descriptor']['digest']
        return '%s@%s' % (name, digest)

    def get_networks_names_ids(self):
        return dict(((network['Name'], network['Id']) for network in self.client.networks()))

    def get_missing_secret_ids(self):
        """
        Resolve missing secret ids by looking them up by name
        """
        secret_names = [secret['secret_name'] for secret in self.client.module.params.get('secrets') or [] if secret['secret_id'] is None]
        if not secret_names:
            return {}
        secrets = self.client.secrets(filters={'name': secret_names})
        secrets = dict(((secret['Spec']['Name'], secret['ID']) for secret in secrets if secret['Spec']['Name'] in secret_names))
        for secret_name in secret_names:
            if secret_name not in secrets:
                self.client.fail('Could not find a secret named "%s"' % secret_name)
        return secrets

    def get_missing_config_ids(self):
        """
        Resolve missing config ids by looking them up by name
        """
        config_names = [config['config_name'] for config in self.client.module.params.get('configs') or [] if config['config_id'] is None]
        if not config_names:
            return {}
        configs = self.client.configs(filters={'name': config_names})
        configs = dict(((config['Spec']['Name'], config['ID']) for config in configs if config['Spec']['Name'] in config_names))
        for config_name in config_names:
            if config_name not in configs:
                self.client.fail('Could not find a config named "%s"' % config_name)
        return configs

    def run(self):
        self.diff_tracker = DifferenceTracker()
        module = self.client.module
        image = module.params['image']
        try:
            image_digest = self.get_image_digest(name=image, resolve=module.params['resolve_image'])
        except DockerException as e:
            self.client.fail('Error looking for an image named %s: %s' % (image, to_native(e)))
        try:
            current_service = self.get_service(module.params['name'])
        except Exception as e:
            self.client.fail('Error looking for service named %s: %s' % (module.params['name'], to_native(e)))
        try:
            secret_ids = self.get_missing_secret_ids()
            config_ids = self.get_missing_config_ids()
            network_ids = self.get_networks_names_ids()
            new_service = DockerService.from_ansible_params(module.params, current_service, image_digest, secret_ids, config_ids, network_ids, self.client.docker_api_version, self.client.docker_py_version)
        except Exception as e:
            return self.client.fail('Error parsing module parameters: %s' % to_native(e))
        changed = False
        msg = 'noop'
        rebuilt = False
        differences = DifferenceTracker()
        facts = {}
        if current_service:
            if module.params['state'] == 'absent':
                if not module.check_mode:
                    self.remove_service(module.params['name'])
                msg = 'Service removed'
                changed = True
            else:
                changed, differences, need_rebuild, force_update = new_service.compare(current_service)
                if changed:
                    self.diff_tracker.merge(differences)
                    if need_rebuild:
                        if not module.check_mode:
                            self.remove_service(module.params['name'])
                            self.create_service(module.params['name'], new_service)
                        msg = 'Service rebuilt'
                        rebuilt = True
                    else:
                        if not module.check_mode:
                            self.update_service(module.params['name'], current_service, new_service)
                        msg = 'Service updated'
                        rebuilt = False
                elif force_update:
                    if not module.check_mode:
                        self.update_service(module.params['name'], current_service, new_service)
                    msg = 'Service forcefully updated'
                    rebuilt = False
                    changed = True
                else:
                    msg = 'Service unchanged'
                facts = new_service.get_facts()
        elif module.params['state'] == 'absent':
            msg = 'Service absent'
        else:
            if not module.check_mode:
                self.create_service(module.params['name'], new_service)
            msg = 'Service created'
            changed = True
            facts = new_service.get_facts()
        return (msg, changed, rebuilt, differences.get_legacy_docker_diffs(), facts)

    def run_safe(self):
        while True:
            try:
                return self.run()
            except APIError as e:
                if self.retries > 0 and 'update out of sequence' in str(e.explanation):
                    self.retries -= 1
                    time.sleep(1)
                else:
                    raise