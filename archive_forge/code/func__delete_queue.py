import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients import progress
from heat.engine.resources import stack_user
def _delete_queue(self):
    queue_id = self.data().get('metadata_queue_id')
    if not queue_id:
        return
    endpoint_exists = self.client_plugin().does_endpoint_exist('zaqar', 'messaging')
    if endpoint_exists:
        client_plugin = self.client_plugin('zaqar')
        zaqar = client_plugin.create_for_tenant(self.stack.stack_user_project_id, self._user_token())
        with client_plugin.ignore_not_found:
            zaqar.queue(queue_id).delete()
    self.data_delete('metadata_queue_id')