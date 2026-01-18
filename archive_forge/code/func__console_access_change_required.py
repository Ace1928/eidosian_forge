import secrets
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def _console_access_change_required(user_response, requested_boolean):
    current_boolean = CREATE_DEFAULTS['console_access']
    if 'ConsoleAccess' in user_response:
        current_boolean = user_response['ConsoleAccess']
    elif 'Pending' in user_response:
        current_boolean = user_response['Pending']['ConsoleAccess']
    return current_boolean != requested_boolean