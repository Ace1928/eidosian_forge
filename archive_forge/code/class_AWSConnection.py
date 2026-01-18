import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto3_conn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_aws_connection_info
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
class AWSConnection:
    """
    Create the connection object and client objects as required.
    """

    def __init__(self, ansible_obj, resources, use_boto3=True):
        try:
            self.region, self.endpoint, aws_connect_kwargs = get_aws_connection_info(ansible_obj, boto3=use_boto3)
            self.resource_client = dict()
            if not resources:
                resources = ['lambda']
            resources.append('iam')
            for resource in resources:
                aws_connect_kwargs.update(dict(region=self.region, endpoint=self.endpoint, conn_type='client', resource=resource))
                self.resource_client[resource] = boto3_conn(ansible_obj, **aws_connect_kwargs)
            if not self.region:
                self.region = self.resource_client['lambda'].meta.region_name
        except (ClientError, ParamValidationError, MissingParametersError) as e:
            ansible_obj.fail_json(msg=f'Unable to connect, authorize or access resource: {e}')
        try:
            self.account_id = self.resource_client['iam'].get_user()['User']['Arn'].split(':')[4]
        except (ClientError, ValueError, KeyError, IndexError):
            self.account_id = ''

    def client(self, resource='lambda'):
        return self.resource_client[resource]