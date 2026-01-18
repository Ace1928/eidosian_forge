from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def _get_failed_deployment_operations(self, name):
    results = []
    try:
        operations = self.rm_client.deployment_operations.list(self.resource_group, name)
    except Exception as exc:
        self.fail('Get deployment failed with status code: %s and message: %s' % (exc.status_code, exc.message))
    try:
        results = [dict(id=op.id, operation_id=op.operation_id, status_code=op.properties.status_code, status_message=op.properties.status_message, target_resource=dict(id=op.properties.target_resource.id, resource_name=op.properties.target_resource.resource_name, resource_type=op.properties.target_resource.resource_type) if op.properties.target_resource else None, provisioning_state=op.properties.provisioning_state) for op in self._get_failed_nested_operations(operations)]
    except Exception:
        pass
    self.log(dict(failed_deployment_operations=results), pretty_print=True)
    return results