import base64
import re  # regex library
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.acm import ACMServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def PEM_body_to_DER(module, pem):
    try:
        der = base64.b64decode(to_text(pem))
    except (ValueError, TypeError) as e:
        module.fail_json_aws(e, msg='Unable to decode certificate chain')
    return der