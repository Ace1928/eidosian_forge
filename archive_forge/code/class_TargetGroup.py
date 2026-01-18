from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class TargetGroup(object):
    """Models an elbv2 target group"""

    def __init__(self, **kwargs):
        self.target_group_type = kwargs['target_group_type']
        self.target_group_arn = kwargs['target_group_arn']
        self.targets = []

    def add_target(self, target_id, target_port, target_az, raw_target_health):
        self.targets.append(Target(target_id, target_port, target_az, raw_target_health))

    def to_dict(self):
        object_dict = vars(self)
        object_dict['targets'] = [vars(each) for each in self.get_targets()]
        return object_dict

    def get_targets(self):
        return list(self.targets)