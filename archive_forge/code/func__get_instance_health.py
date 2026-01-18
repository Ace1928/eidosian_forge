from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_instance_health(self, lb):
    """
        Check instance health, should return status object or None under
        certain error conditions.
        """
    try:
        status = self.client_elb.describe_instance_health(aws_retry=True, LoadBalancerName=lb['LoadBalancerName'], Instances=[{'InstanceId': self.instance_id}])['InstanceStates']
    except is_boto3_error_code('InvalidInstance'):
        return None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Failed to get instance health')
    if not status:
        return None
    return status[0]['State']