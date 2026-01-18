from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import convert_managed_policy_names_to_arns
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_group
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_group
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def add_group_members(connection, module, group_name, members):
    for user in members:
        IAMErrorHandler.common_error_handler(f'add user {user} to group')(connection.add_user_to_group)(aws_retry=True, GroupName=group_name, UserName=user)