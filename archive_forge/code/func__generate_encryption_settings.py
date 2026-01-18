import os
import getpass
import json
import pty
import random
import re
import select
import string
import subprocess
import time
from functools import wraps
from ansible_collections.amazon.aws.plugins.module_utils.botocore import HAS_BOTO3
from ansible.errors import AnsibleConnectionFailure
from ansible.errors import AnsibleError
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves import xrange
from ansible.module_utils._text import to_bytes
from ansible.module_utils._text import to_text
from ansible.plugins.connection import ConnectionBase
from ansible.plugins.shell.powershell import _common_args
from ansible.utils.display import Display
def _generate_encryption_settings(self):
    put_args = {}
    put_headers = {}
    if not self.get_option('bucket_sse_mode'):
        return (put_args, put_headers)
    put_args['ServerSideEncryption'] = self.get_option('bucket_sse_mode')
    put_headers['x-amz-server-side-encryption'] = self.get_option('bucket_sse_mode')
    if self.get_option('bucket_sse_mode') == 'aws:kms' and self.get_option('bucket_sse_kms_key_id'):
        put_args['SSEKMSKeyId'] = self.get_option('bucket_sse_kms_key_id')
        put_headers['x-amz-server-side-encryption-aws-kms-key-id'] = self.get_option('bucket_sse_kms_key_id')
    return (put_args, put_headers)