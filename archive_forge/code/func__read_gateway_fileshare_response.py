from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@staticmethod
def _read_gateway_fileshare_response(fileshares, aws_reponse):
    for share in aws_reponse['FileShareInfoList']:
        share_obj = camel_dict_to_snake_dict(share)
        if 'gateway_arn' in share_obj:
            del share_obj['gateway_arn']
        fileshares.append(share_obj)
    return aws_reponse['NextMarker'] if 'NextMarker' in aws_reponse else None