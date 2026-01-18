from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_auto_scaling_group_lbs(self):
    """Returns a list of ELBs associated with self.instance_id
        indirectly through its auto scaling group membership"""
    try:
        asg_instances = self.client_asg.describe_auto_scaling_instances(aws_retry=True, InstanceIds=[self.instance_id])['AutoScalingInstances']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Failed to describe ASG Instance')
    if len(asg_instances) > 1:
        self.module.fail_json(msg='Illegal state, expected one auto scaling group instance.')
    if not asg_instances:
        return []
    asg_name = asg_instances[0]['AutoScalingGroupName']
    try:
        asg_instances = self.client_asg.describe_auto_scaling_groups(aws_retry=True, AutoScalingGroupNames=[asg_name])['AutoScalingGroups']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Failed to describe ASG Instance')
    if len(asg_instances) != 1:
        self.module.fail_json(msg='Illegal state, expected one auto scaling group.')
    return asg_instances[0]['LoadBalancerNames']