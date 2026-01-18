from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .botocore import is_boto3_error_code
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
def _match_tags(self, ref_tags, cert_tags):
    if ref_tags is None:
        return True
    try:
        return all((k in cert_tags for k in ref_tags)) and all((cert_tags.get(k) == ref_tags[k] for k in ref_tags))
    except (TypeError, AttributeError) as e:
        self.module.fail_json_aws(e, msg='ACM tag filtering err')