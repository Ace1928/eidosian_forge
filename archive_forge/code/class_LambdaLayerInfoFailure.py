from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
class LambdaLayerInfoFailure(Exception):

    def __init__(self, exc, msg):
        self.exc = exc
        self.msg = msg
        super().__init__(self)