import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients import progress
from heat.engine.resources import stack_user
def _populate_deployments_metadata(self, meta, props):
    meta['deployments'] = meta.get('deployments', [])
    meta['os-collect-config'] = meta.get('os-collect-config', {})
    occ = meta['os-collect-config']
    collectors = list(self.default_collectors)
    occ['collectors'] = collectors
    occ_keys = ('heat', 'zaqar', 'cfn', 'request')
    for occ_key in occ_keys:
        if occ_key not in occ:
            continue
        existing = occ[occ_key]
        for k in existing:
            existing[k] = None
    queue_id = self.data().get('metadata_queue_id')
    if self.transport_poll_server_heat(props):
        occ.update({'heat': {'user_id': self._get_user_id(), 'password': self.password, 'auth_url': self.keystone().server_keystone_endpoint_url(fallback_endpoint=self.context.auth_url), 'project_id': self.stack.stack_user_project_id, 'stack_id': self.stack.identifier().stack_path(), 'resource_name': self.name, 'region_name': self._get_region_name()}})
        collectors.append('heat')
    elif self.transport_zaqar_message(props):
        queue_id = queue_id or self.physical_resource_name()
        occ.update({'zaqar': {'user_id': self._get_user_id(), 'password': self.password, 'auth_url': self.keystone().server_keystone_endpoint_url(fallback_endpoint=self.context.auth_url), 'project_id': self.stack.stack_user_project_id, 'queue_id': queue_id, 'region_name': self._get_region_name()}})
        collectors.append('zaqar')
    elif self.transport_poll_server_cfn(props):
        heat_client_plugin = self.stack.clients.client_plugin('heat')
        config_url = heat_client_plugin.get_cfn_metadata_server_url()
        occ.update({'cfn': {'metadata_url': config_url, 'access_key_id': self.access_key, 'secret_access_key': self.secret_key, 'stack_name': self.stack.name, 'path': '%s.Metadata' % self.name}})
        collectors.append('cfn')
    elif self.transport_poll_temp_url(props):
        container_name, object_name = self._container_and_object_name(props)
        self.client('swift').put_container(container_name)
        url = self.client_plugin('swift').get_temp_url(container_name, object_name, method='GET')
        put_url = self.client_plugin('swift').get_temp_url(container_name, object_name)
        self.data_set('metadata_put_url', put_url)
        self.data_set('metadata_object_name', object_name)
        collectors.append('request')
        occ.update({'request': {'metadata_url': url}})
    collectors.append('local')
    self.metadata_set(meta)
    if queue_id:
        zaqar_plugin = self.client_plugin('zaqar')
        zaqar = zaqar_plugin.create_for_tenant(self.stack.stack_user_project_id, self._user_token())
        queue = zaqar.queue(queue_id)
        queue.post({'body': meta, 'ttl': zaqar_plugin.DEFAULT_TTL})
        self.data_set('metadata_queue_id', queue_id)
    object_name = self.data().get('metadata_object_name')
    if object_name:
        container_name, object_name = self._container_and_object_name(props)
        self.client('swift').put_object(container_name, object_name, jsonutils.dumps(meta))