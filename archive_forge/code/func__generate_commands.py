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
def _generate_commands(self, bucket_name, s3_path, in_path, out_path):
    put_args, put_headers = self._generate_encryption_settings()
    put_url = self._get_url('put_object', bucket_name, s3_path, 'PUT', extra_args=put_args)
    get_url = self._get_url('get_object', bucket_name, s3_path, 'GET')
    if self.is_windows:
        put_command_headers = '; '.join([f"'{h}' = '{v}'" for h, v in put_headers.items()])
        put_commands = [f"Invoke-WebRequest -Method PUT -Headers @{{{put_command_headers}}} -InFile '{in_path}' -Uri '{put_url}' -UseBasicParsing"]
        get_commands = [f"Invoke-WebRequest '{get_url}' -OutFile '{out_path}'"]
    else:
        put_command_headers = ' '.join([f"-H '{h}: {v}'" for h, v in put_headers.items()])
        put_commands = [f"curl --request PUT {put_command_headers} --upload-file '{in_path}' '{put_url}'"]
        get_commands = [f"curl -o '{out_path}' '{get_url}'", f"touch '{out_path}'"]
    return (get_commands, put_commands, put_args)