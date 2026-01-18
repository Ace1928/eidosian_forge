import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients import progress
from heat.engine.resources import stack_user
def _container_and_object_name(self, props):
    deployment_swift_data = props.get(self.DEPLOYMENT_SWIFT_DATA, self.properties[self.DEPLOYMENT_SWIFT_DATA])
    container_name = deployment_swift_data[self.CONTAINER]
    if container_name is None:
        container_name = self.physical_resource_name()
    object_name = deployment_swift_data[self.OBJECT]
    if object_name is None:
        object_name = self.data().get('metadata_object_name')
    if object_name is None:
        object_name = str(uuid.uuid4())
    return (container_name, object_name)