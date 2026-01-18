import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def _targets_to_put(self):
    """Returns a list of targets that need to be updated or added remotely"""
    remote_targets = self.rule.list_targets()
    temp = []
    for t in self.targets:
        if t['input_transformer'] is not None and t['input_transformer']['input_template'] is not None:
            val = t['input_transformer']['input_template']
            valid_json = _validate_json(val)
            if not valid_json:
                t['input_transformer']['input_template'] = '"' + val + '"'
        temp.append(scrub_none_parameters(t))
    self.targets = temp
    return [t for t in self.targets if camel_dict_to_snake_dict(t) not in remote_targets]