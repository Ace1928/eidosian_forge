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
def _get_boto_client(self, service, region_name=None, profile_name=None, endpoint_url=None):
    """Gets a boto3 client based on the STS token"""
    aws_access_key_id = self.get_option('access_key_id')
    aws_secret_access_key = self.get_option('secret_access_key')
    aws_session_token = self.get_option('session_token')
    session_args = dict(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token, region_name=region_name)
    if profile_name:
        session_args['profile_name'] = profile_name
    session = boto3.session.Session(**session_args)
    client = session.client(service, endpoint_url=endpoint_url, config=Config(signature_version='s3v4', s3={'addressing_style': self.get_option('s3_addressing_style')}))
    return client