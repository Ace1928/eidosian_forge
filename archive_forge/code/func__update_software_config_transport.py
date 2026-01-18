import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients import progress
from heat.engine.resources import stack_user
def _update_software_config_transport(self, prop_diff):
    if not self.user_data_software_config():
        return
    self._delete_queue()
    self._delete_temp_url()
    metadata = self.metadata_get(True) or {}
    self._create_transport_credentials(prop_diff)
    self._populate_deployments_metadata(metadata, prop_diff)