import copy
import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _await_glue_connection(connection, module):
    start_time = time.time()
    wait_timeout = start_time + 30
    check_interval = 5
    while wait_timeout > time.time():
        glue_connection = _get_glue_connection(connection, module)
        if glue_connection and glue_connection.get('Name'):
            return glue_connection
        time.sleep(check_interval)
    module.fail_json(msg=f'Timeout waiting for Glue connection {module.params.get('name')}')