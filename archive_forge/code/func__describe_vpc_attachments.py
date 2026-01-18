from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
@Boto3Mixin.aws_error_handler('describe transit gateway attachments')
def _describe_vpc_attachments(self, **params):
    result = self._paginated_describe_transit_gateway_vpc_attachments(**params)
    return result.get('TransitGatewayVpcAttachments', None)