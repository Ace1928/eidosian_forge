from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _create_metrics_configuration(mc_id, filter_prefix, filter_tags):
    payload = {'Id': mc_id}
    if filter_prefix and (not filter_tags):
        payload['Filter'] = {'Prefix': filter_prefix}
    elif not filter_prefix and len(filter_tags) == 1:
        payload['Filter'] = {'Tag': ansible_dict_to_boto3_tag_list(filter_tags)[0]}
    elif filter_tags:
        payload['Filter'] = {'And': {'Tags': ansible_dict_to_boto3_tag_list(filter_tags)}}
        if filter_prefix:
            payload['Filter']['And']['Prefix'] = filter_prefix
    return payload