from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_target_descriptions(self, target_groups):
    """Helper function to build a list of all the target descriptions
        for this target in a target group"""
    tgs = set()
    for tg in target_groups:
        try:
            response = self.elbv2.describe_target_health(TargetGroupArn=tg.target_group_arn, aws_retry=True)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg='Could not describe target ' + f'health for target group {tg.target_group_arn}')
        for t in response['TargetHealthDescriptions']:
            if t['Target']['Id'] == self.instance_id or t['Target']['Id'] in self.instance_ips:
                az = t['Target']['AvailabilityZone'] if 'AvailabilityZone' in t['Target'] else None
                tg.add_target(t['Target']['Id'], t['Target']['Port'], az, t['TargetHealth'])
                tgs.add(tg)
    return list(tgs)