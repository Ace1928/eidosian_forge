from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_target_groups(self):
    self.instance_ips = self._get_instance_ips()
    target_groups = self._get_target_group_objects()
    return self._get_target_descriptions(target_groups)