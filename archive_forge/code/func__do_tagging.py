from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.base import BaseResourceManager
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.base import Boto3Mixin
def _do_tagging(self):
    changed = False
    tags_to_add = self._tagging_updates.get('add')
    tags_to_remove = self._tagging_updates.get('remove')
    if tags_to_add:
        changed = True
        tags = ansible_dict_to_boto3_tag_list(tags_to_add)
        if not self.module.check_mode:
            self._add_tags(Resources=[self.resource_id], Tags=tags)
    if tags_to_remove:
        changed = True
        if not self.module.check_mode:
            tag_list = [dict(Key=tagkey) for tagkey in tags_to_remove]
            self._remove_tags(Resources=[self.resource_id], Tags=tag_list)
    return changed