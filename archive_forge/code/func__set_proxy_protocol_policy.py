from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def _set_proxy_protocol_policy(self, policy_name):
    """Install a proxy protocol policy if needed"""
    policy_map = self._get_policy_map()
    policy_attributes = [dict(AttributeName='ProxyProtocol', AttributeValue='true')]
    proxy_policy = dict(PolicyName=policy_name, PolicyTypeName='ProxyProtocolPolicyType', PolicyAttributeDescriptions=policy_attributes)
    existing_policy = policy_map.get(policy_name)
    if proxy_policy == existing_policy:
        return False
    if existing_policy is not None:
        self.module.fail_json(msg=f"Unable to configure ProxyProtocol policy. Policy with name {policy_name} already exists and doesn't match.", policy=proxy_policy, existing_policy=existing_policy)
    proxy_policy['PolicyAttributes'] = proxy_policy.pop('PolicyAttributeDescriptions')
    proxy_policy['LoadBalancerName'] = self.name
    self.changed = True
    if self.check_mode:
        return True
    try:
        self.client.create_load_balancer_policy(aws_retry=True, **proxy_policy)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg='Failed to create load balancer policy', policy=proxy_policy)
    return True