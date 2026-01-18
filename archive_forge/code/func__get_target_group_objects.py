from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_target_group_objects(self):
    """helper function to build a list of TargetGroup objects based on
        the AWS API"""
    try:
        paginator = self.elbv2.get_paginator('describe_target_groups')
        tg_response = paginator.paginate().build_full_result()
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e, msg='Could not describe target groups')
    target_groups = []
    for each_tg in tg_response['TargetGroups']:
        if not self.get_unused_target_groups and len(each_tg['LoadBalancerArns']) < 1:
            continue
        target_groups.append(TargetGroup(target_group_arn=each_tg['TargetGroupArn'], target_group_type=each_tg['TargetType']))
    return target_groups