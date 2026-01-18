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
def _wrap_command(self, cmd, sudoable, mark_start, mark_end):
    """wrap command so stdout and status can be extracted"""
    if self.is_windows:
        if not cmd.startswith(' '.join(_common_args) + ' -EncodedCommand'):
            cmd = self._shell._encode_script(cmd, preserve_rc=True)
        cmd = cmd + '; echo ' + mark_start + '\necho ' + mark_end + '\n'
    else:
        cmd = f"""printf '%s\\n' '{mark_start}';\necho | {cmd};\nprintf '\\n%s\\n%s\\n' "$?" '{mark_end}';\n"""
    self._vvvv(f"_wrap_command: \n'{to_text(cmd)}'")
    return cmd