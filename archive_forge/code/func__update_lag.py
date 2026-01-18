import time
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_connection
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import disassociate_connection_and_lag
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(retries=5, delay=2, backoff=2.0, catch_extra_error_codes=['DirectConnectClientException'])
def _update_lag(client, lag_id, lag_name, min_links):
    params = {}
    if min_links:
        params.update(minimumLinks=min_links)
    if lag_name:
        params.update(lagName=lag_name)
    client.update_lag(lagId=lag_id, **params)