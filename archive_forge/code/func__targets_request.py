import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def _targets_request(self, targets):
    """Formats each target for the request"""
    targets_request = []
    for target in targets:
        target_request = scrub_none_parameters(snake_dict_to_camel_dict(target, True))
        if target_request.get('Input', None):
            target_request['Input'] = _format_json(target_request['Input'])
        if target_request.get('InputTransformer', None):
            if target_request.get('InputTransformer').get('InputTemplate', None):
                target_request['InputTransformer']['InputTemplate'] = _format_json(target_request['InputTransformer']['InputTemplate'])
            if target_request.get('InputTransformer').get('InputPathsMap', None):
                target_request['InputTransformer']['InputPathsMap'] = target['input_transformer']['input_paths_map']
        targets_request.append(target_request)
    return targets_request