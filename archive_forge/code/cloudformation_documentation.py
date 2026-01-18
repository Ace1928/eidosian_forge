import json
import time
import traceback
import uuid
from hashlib import sha1
from ansible.module_utils._text import to_bytes
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
Create a change set, describe it and delete it before returning check mode outputs.