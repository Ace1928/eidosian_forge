import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class CloudFrontOriginAccessIdentityServiceManager(object):
    """
    Handles CloudFront origin access identity service calls to aws
    """

    def __init__(self, module):
        self.module = module
        self.client = module.client('cloudfront')

    def create_origin_access_identity(self, caller_reference, comment):
        try:
            return self.client.create_cloud_front_origin_access_identity(CloudFrontOriginAccessIdentityConfig={'CallerReference': caller_reference, 'Comment': comment})
        except (ClientError, BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Error creating cloud front origin access identity.')

    def delete_origin_access_identity(self, origin_access_identity_id, e_tag):
        try:
            result = self.client.delete_cloud_front_origin_access_identity(Id=origin_access_identity_id, IfMatch=e_tag)
            return (result, True)
        except (ClientError, BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Error deleting Origin Access Identity.')

    def update_origin_access_identity(self, caller_reference, comment, origin_access_identity_id, e_tag):
        changed = False
        new_config = {'CallerReference': caller_reference, 'Comment': comment}
        try:
            current_config = self.client.get_cloud_front_origin_access_identity_config(Id=origin_access_identity_id)['CloudFrontOriginAccessIdentityConfig']
        except (ClientError, BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Error getting Origin Access Identity config.')
        if new_config != current_config:
            changed = True
        try:
            result = self.client.update_cloud_front_origin_access_identity(CloudFrontOriginAccessIdentityConfig=new_config, Id=origin_access_identity_id, IfMatch=e_tag)
        except (ClientError, BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Error updating Origin Access Identity.')
        return (result, changed)