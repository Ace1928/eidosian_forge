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
def _exec_transport_commands(self, in_path, out_path, commands):
    stdout_combined, stderr_combined = ('', '')
    for command in commands:
        returncode, stdout, stderr = self.exec_command(command, in_data=None, sudoable=False)
        if returncode != 0:
            raise AnsibleError(f'failed to transfer file to {in_path} {out_path}:\n{stdout}\n{stderr}')
        stdout_combined += stdout
        stderr_combined += stderr
    return (returncode, stdout_combined, stderr_combined)