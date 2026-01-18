import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def _sync_state(self, enabled=True):
    """Syncs local rule state with AWS"""
    remote_state = self._remote_state()
    if enabled and remote_state != 'ENABLED':
        self.rule.enable()
    elif not enabled and remote_state != 'DISABLED':
        self.rule.disable()