import json
import os
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import text_type
from .common import get_collection_info
from .exceptions import AnsibleBotocoreError
from .retries import AWSRetry
def _aws_region(params):
    region = params.get('region')
    if region:
        return region
    if not HAS_BOTO3:
        raise AnsibleBotocoreError(message=missing_required_lib('boto3 and botocore'), exception=BOTO3_IMP_ERR)
    try:
        profile_name = params.get('profile') or None
        return botocore.session.Session(profile=profile_name).get_config_variable('region')
    except botocore.exceptions.ProfileNotFound:
        return None