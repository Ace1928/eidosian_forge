import secrets
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def _group_change_required(user_response, requested_groups):
    current_groups = []
    if 'Groups' in user_response:
        current_groups = user_response['Groups']
    elif 'Pending' in user_response:
        current_groups = user_response['Pending']['Groups']
    if len(current_groups) != len(requested_groups):
        return True
    if len(current_groups) != len(set(current_groups) & set(requested_groups)):
        return True
    return False