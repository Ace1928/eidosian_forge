from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
def _do_create_resource(self):
    params = self._merge_resource_changes(filter_immutable=False, creation=True)
    response = self._create_vpc_attachment(**params)
    if response:
        self.resource_id = response.get('TransitGatewayAttachmentId', None)
    return response