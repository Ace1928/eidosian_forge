from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class EcsAttributes(object):
    """Handles ECS Cluster Attribute"""

    def __init__(self, module, attributes):
        self.module = module
        self.attributes = attributes if self._validate_attrs(attributes) else self._parse_attrs(attributes)

    def __bool__(self):
        return bool(self.attributes)
    __nonzero__ = __bool__

    def __iter__(self):
        return iter(self.attributes)

    @staticmethod
    def _validate_attrs(attrs):
        return all((tuple(attr.keys()) in (('name', 'value'), ('value', 'name')) for attr in attrs))

    def _parse_attrs(self, attrs):
        attrs_parsed = []
        for attr in attrs:
            if isinstance(attr, dict):
                if len(attr) != 1:
                    self.module.fail_json(msg=f'Incorrect attribute format - {str(attr)}')
                name, value = list(attr.items())[0]
                attrs_parsed.append({'name': name, 'value': value})
            elif isinstance(attr, str):
                attrs_parsed.append({'name': attr, 'value': None})
            else:
                self.module.fail_json(msg=f'Incorrect attributes format - {str(attrs)}')
        return attrs_parsed

    def _setup_attr_obj(self, ecs_arn, name, value=None, skip_value=False):
        attr_obj = {'targetType': 'container-instance', 'targetId': ecs_arn, 'name': name}
        if not skip_value and value is not None:
            attr_obj['value'] = value
        return attr_obj

    def get_for_ecs_arn(self, ecs_arn, skip_value=False):
        """
        Returns list of attribute dicts ready to be passed to boto3
        attributes put/delete methods.
        """
        return [self._setup_attr_obj(ecs_arn, skip_value=skip_value, **attr) for attr in self.attributes]

    def diff(self, attrs):
        """
        Returns EcsAttributes Object containing attributes which are present
        in self but are absent in passed attrs (EcsAttributes Object).
        """
        attrs_diff = [attr for attr in self.attributes if attr not in attrs]
        return EcsAttributes(self.module, attrs_diff)