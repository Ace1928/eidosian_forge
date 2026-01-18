import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients import progress
from heat.engine.resources import stack_user
def _delete_temp_url(self):
    object_name = self.data().get('metadata_object_name')
    if not object_name:
        return
    endpoint_exists = self.client_plugin().does_endpoint_exist('swift', 'object-store')
    if endpoint_exists:
        with self.client_plugin('swift').ignore_not_found:
            container = self.properties[self.DEPLOYMENT_SWIFT_DATA].get('container')
            container = container or self.physical_resource_name()
            swift = self.client('swift')
            swift.delete_object(container, object_name)
            headers = swift.head_container(container)
            if int(headers['x-container-object-count']) == 0:
                swift.delete_container(container)
    self.data_delete('metadata_object_name')
    self.data_delete('metadata_put_url')