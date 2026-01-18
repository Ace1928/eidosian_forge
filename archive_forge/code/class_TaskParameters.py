from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible.module_utils.common.text.converters import to_native
class TaskParameters(DockerBaseClass):

    def __init__(self):
        super(TaskParameters, self).__init__()
        self.advertise_addr = None
        self.listen_addr = None
        self.remote_addrs = None
        self.join_token = None
        self.data_path_addr = None
        self.data_path_port = None
        self.snapshot_interval = None
        self.task_history_retention_limit = None
        self.keep_old_snapshots = None
        self.log_entries_for_slow_followers = None
        self.heartbeat_tick = None
        self.election_tick = None
        self.dispatcher_heartbeat_period = None
        self.node_cert_expiry = None
        self.name = None
        self.labels = None
        self.log_driver = None
        self.signing_ca_cert = None
        self.signing_ca_key = None
        self.ca_force_rotate = None
        self.autolock_managers = None
        self.rotate_worker_token = None
        self.rotate_manager_token = None
        self.default_addr_pool = None
        self.subnet_size = None

    @staticmethod
    def from_ansible_params(client):
        result = TaskParameters()
        for key, value in client.module.params.items():
            if key in result.__dict__:
                setattr(result, key, value)
        result.update_parameters(client)
        return result

    def update_from_swarm_info(self, swarm_info):
        spec = swarm_info['Spec']
        ca_config = spec.get('CAConfig') or dict()
        if self.node_cert_expiry is None:
            self.node_cert_expiry = ca_config.get('NodeCertExpiry')
        if self.ca_force_rotate is None:
            self.ca_force_rotate = ca_config.get('ForceRotate')
        dispatcher = spec.get('Dispatcher') or dict()
        if self.dispatcher_heartbeat_period is None:
            self.dispatcher_heartbeat_period = dispatcher.get('HeartbeatPeriod')
        raft = spec.get('Raft') or dict()
        if self.snapshot_interval is None:
            self.snapshot_interval = raft.get('SnapshotInterval')
        if self.keep_old_snapshots is None:
            self.keep_old_snapshots = raft.get('KeepOldSnapshots')
        if self.heartbeat_tick is None:
            self.heartbeat_tick = raft.get('HeartbeatTick')
        if self.log_entries_for_slow_followers is None:
            self.log_entries_for_slow_followers = raft.get('LogEntriesForSlowFollowers')
        if self.election_tick is None:
            self.election_tick = raft.get('ElectionTick')
        orchestration = spec.get('Orchestration') or dict()
        if self.task_history_retention_limit is None:
            self.task_history_retention_limit = orchestration.get('TaskHistoryRetentionLimit')
        encryption_config = spec.get('EncryptionConfig') or dict()
        if self.autolock_managers is None:
            self.autolock_managers = encryption_config.get('AutoLockManagers')
        if self.name is None:
            self.name = spec['Name']
        if self.labels is None:
            self.labels = spec.get('Labels') or {}
        if 'LogDriver' in spec['TaskDefaults']:
            self.log_driver = spec['TaskDefaults']['LogDriver']

    def update_parameters(self, client):
        assign = dict(snapshot_interval='snapshot_interval', task_history_retention_limit='task_history_retention_limit', keep_old_snapshots='keep_old_snapshots', log_entries_for_slow_followers='log_entries_for_slow_followers', heartbeat_tick='heartbeat_tick', election_tick='election_tick', dispatcher_heartbeat_period='dispatcher_heartbeat_period', node_cert_expiry='node_cert_expiry', name='name', labels='labels', signing_ca_cert='signing_ca_cert', signing_ca_key='signing_ca_key', ca_force_rotate='ca_force_rotate', autolock_managers='autolock_managers', log_driver='log_driver')
        params = dict()
        for dest, source in assign.items():
            if not client.option_minimal_versions[source]['supported']:
                continue
            value = getattr(self, source)
            if value is not None:
                params[dest] = value
        self.spec = client.create_swarm_spec(**params)

    def compare_to_active(self, other, client, differences):
        for k in self.__dict__:
            if k in ('advertise_addr', 'listen_addr', 'remote_addrs', 'join_token', 'rotate_worker_token', 'rotate_manager_token', 'spec', 'default_addr_pool', 'subnet_size', 'data_path_addr', 'data_path_port'):
                continue
            if not client.option_minimal_versions[k]['supported']:
                continue
            value = getattr(self, k)
            if value is None:
                continue
            other_value = getattr(other, k)
            if value != other_value:
                differences.add(k, parameter=value, active=other_value)
        if self.rotate_worker_token:
            differences.add('rotate_worker_token', parameter=True, active=False)
        if self.rotate_manager_token:
            differences.add('rotate_manager_token', parameter=True, active=False)
        return differences