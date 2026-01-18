from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _enable_availailability_zone(self, lb):
    """Enable the current instance's availability zone in the provided lb.
        Returns True if the zone was enabled or False if no change was made.
        lb: load balancer"""
    instance = self._get_instance()
    desired_zone = instance['Placement']['AvailabilityZone']
    if desired_zone in lb['AvailabilityZones']:
        return False
    if self.module.check_mode:
        return True
    try:
        self.client_elb.enable_availability_zones_for_load_balancer(aws_retry=True, LoadBalancerName=lb['LoadBalancerName'], AvailabilityZones=[desired_zone])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, 'Failed to enable AZ on load balancers', load_balancer=lb, zone=desired_zone)
    return True