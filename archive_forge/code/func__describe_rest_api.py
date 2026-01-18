from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
@AWSRetry.jittered_backoff()
def _describe_rest_api(connection, module, rest_api_id):
    try:
        response = connection.get_rest_api(restApiId=rest_api_id)
        response.pop('ResponseMetadata')
    except is_boto3_error_code('ResourceNotFoundException'):
        response = {}
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f"Trying to get Rest API '{rest_api_id}'.")
    return response