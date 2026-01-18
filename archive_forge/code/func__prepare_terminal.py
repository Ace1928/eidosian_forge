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
def _prepare_terminal(self):
    """perform any one-time terminal settings"""
    if self.is_windows:
        return
    startup_complete = False
    disable_echo_complete = None
    disable_echo_cmd = to_bytes('stty -echo\n', errors='surrogate_or_strict')
    disable_prompt_complete = None
    end_mark = ''.join([random.choice(string.ascii_letters) for i in xrange(self.MARK_LENGTH)])
    disable_prompt_cmd = to_bytes("PS1='' ; bind 'set enable-bracketed-paste off'; printf '\\n%s\\n' '" + end_mark + "'\n", errors='surrogate_or_strict')
    disable_prompt_reply = re.compile('\\r\\r\\n' + re.escape(end_mark) + '\\r\\r\\n', re.MULTILINE)
    stdout = ''
    stop_time = int(round(time.time())) + self.get_option('ssm_timeout')
    while not disable_prompt_complete and self._session.poll() is None:
        remaining = stop_time - int(round(time.time()))
        if remaining < 1:
            self._timeout = True
            self._vvvv(f'PRE timeout stdout: \n{to_bytes(stdout)}')
            raise AnsibleConnectionFailure(f'SSM start_session timeout on host: {self.instance_id}')
        if self._poll_stdout.poll(1000):
            stdout += to_text(self._stdout.read(1024))
            self._vvvv(f'PRE stdout line: \n{to_bytes(stdout)}')
        else:
            self._vvvv(f'PRE remaining: {remaining}')
        if startup_complete is False:
            match = str(stdout).find('Starting session with SessionId')
            if match != -1:
                self._vvvv('PRE startup output received')
                startup_complete = True
        if startup_complete and disable_echo_complete is None:
            self._vvvv(f'PRE Disabling Echo: {disable_echo_cmd}')
            self._session.stdin.write(disable_echo_cmd)
            disable_echo_complete = False
        if disable_echo_complete is False:
            match = str(stdout).find('stty -echo')
            if match != -1:
                disable_echo_complete = True
        if disable_echo_complete and disable_prompt_complete is None:
            self._vvvv(f'PRE Disabling Prompt: \n{disable_prompt_cmd}')
            self._session.stdin.write(disable_prompt_cmd)
            disable_prompt_complete = False
        if disable_prompt_complete is False:
            match = disable_prompt_reply.search(stdout)
            if match:
                stdout = stdout[match.end():]
                disable_prompt_complete = True
    if not disable_prompt_complete:
        raise AnsibleConnectionFailure(f'SSM process closed during _prepare_terminal on host: {self.instance_id}')
    self._vvvv('PRE Terminal configured')