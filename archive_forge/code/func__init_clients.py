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
def _init_clients(self):
    self._vvvv('INITIALIZE BOTO3 CLIENTS')
    profile_name = self.get_option('profile') or ''
    region_name = self.get_option('region')
    self._vvvv('SETUP BOTO3 CLIENTS: SSM')
    ssm_client = self._get_boto_client('ssm', region_name=region_name, profile_name=profile_name)
    self._client = ssm_client
    s3_endpoint_url, s3_region_name = self._get_bucket_endpoint()
    self._vvvv(f'SETUP BOTO3 CLIENTS: S3 {s3_endpoint_url}')
    s3_bucket_client = self._get_boto_client('s3', region_name=s3_region_name, endpoint_url=s3_endpoint_url, profile_name=profile_name)
    self._s3_client = s3_bucket_client