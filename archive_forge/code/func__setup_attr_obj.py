from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _setup_attr_obj(self, ecs_arn, name, value=None, skip_value=False):
    attr_obj = {'targetType': 'container-instance', 'targetId': ecs_arn, 'name': name}
    if not skip_value and value is not None:
        attr_obj['value'] = value
    return attr_obj