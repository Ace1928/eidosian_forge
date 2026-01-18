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
def _wait_for_instance_state(self, waiter_name, instances):
    if not instances:
        return False
    if self.check_mode:
        return True
    waiter = get_waiter(self.client, waiter_name)
    instance_list = list((dict(InstanceId=instance) for instance in instances))
    try:
        waiter.wait(WaiterConfig=self._waiter_config, LoadBalancerName=self.name, Instances=instance_list)
    except botocore.exceptions.WaiterError as e:
        self.module.fail_json_aws(e, 'Timeout waiting for ELB Instance State')
    return True