from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
def _filter_immutable_resource_attributes(self, resource):
    resource = super(TransitGatewayVpcAttachmentManager, self)._filter_immutable_resource_attributes(resource)
    resource.pop('TransitGatewayId', None)
    resource.pop('VpcId', None)
    resource.pop('VpcOwnerId', None)
    resource.pop('State', None)
    resource.pop('SubnetIds', None)
    resource.pop('CreationTime', None)
    resource.pop('Tags', None)
    return resource