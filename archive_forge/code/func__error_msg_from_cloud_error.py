from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def _error_msg_from_cloud_error(self, exc):
    msg = ''
    status_code = str(exc.status_code)
    if status_code.startswith('2'):
        msg = 'Deployment failed: {0}'.format(exc.message)
    else:
        msg = 'Deployment failed with status code: {0} and message: {1}'.format(status_code, exc.message)
    return msg