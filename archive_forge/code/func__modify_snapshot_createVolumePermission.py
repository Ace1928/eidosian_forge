import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def _modify_snapshot_createVolumePermission(module, ec2, snapshot_id, purge_create_vol_permission):
    update_needed = check_user_or_group_update_needed(module, ec2)
    if not update_needed:
        module.exit_json(changed=False, msg='Supplied CreateVolumePermission already applied, update not needed')
    if purge_create_vol_permission is True:
        _reset_snapshpot_attribute(module, ec2, snapshot_id)
        if not module.params.get('user_ids') and (not module.params.get('group_names')):
            module.exit_json(changed=True, msg='Reset createVolumePermission successfully')
    params = build_modify_createVolumePermission_params(module)
    if module.check_mode:
        module.exit_json(changed=True, msg='Would have modified CreateVolumePermission')
    try:
        ec2.modify_snapshot_attribute(**params)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to modify createVolumePermission')
    module.exit_json(changed=True, msg='Successfully modified CreateVolumePermission')