from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_instance_ips(self):
    """Fetch all IPs associated with this instance so that we can determine
        whether or not an instance is in an IP-based target group"""
    try:
        reservations = self.ec2.describe_instances(InstanceIds=[self.instance_id], aws_retry=True)['Reservations']
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e, msg=f"Could not get instance info for instance '{self.instance_id}'")
    if len(reservations) < 1:
        self.module.fail_json(msg=f'Instance ID {self.instance_id} could not be found')
    instance = reservations[0]['Instances'][0]
    ips = set()
    ips.add(instance['PrivateIpAddress'])
    for nic in instance['NetworkInterfaces']:
        ips.add(nic['PrivateIpAddress'])
        for ip in nic['PrivateIpAddresses']:
            ips.add(ip['PrivateIpAddress'])
    return list(ips)