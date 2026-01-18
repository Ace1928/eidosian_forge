from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_instance_lbs(self, ec2_elbs=None):
    """Returns a list of ELBs attached to self.instance_id
        ec2_elbs: an optional list of elb names that will be used
                  for elb lookup instead of returning what elbs
                  are attached to self.instance_id"""
    list_params = dict()
    if not ec2_elbs:
        ec2_elbs = self._get_auto_scaling_group_lbs()
    if ec2_elbs:
        list_params['LoadBalancerNames'] = ec2_elbs
    try:
        elbs = self._describe_elbs(**list_params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, 'Failed to describe load balancers')
    if ec2_elbs:
        return elbs
    lbs = []
    for lb in elbs:
        instance_ids = [i['InstanceId'] for i in lb['Instances']]
        if self.instance_id in instance_ids:
            lbs.append(lb)
    return lbs