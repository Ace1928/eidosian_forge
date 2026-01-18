from time import sleep
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_nodes_to_remove(self):
    """If there are nodes to remove, it figures out which need to be removed"""
    num_nodes_to_remove = self.data['NumCacheNodes'] - self.num_nodes
    if num_nodes_to_remove <= 0:
        return []
    if not self.hard_modify:
        self.module.fail_json(msg=f"'{self.name}' requires removal of cache nodes. 'hard_modify' must be set to true to proceed.")
    cache_node_ids = [cn['CacheNodeId'] for cn in self.data['CacheNodes']]
    return cache_node_ids[-num_nodes_to_remove:]