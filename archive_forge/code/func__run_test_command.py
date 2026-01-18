import datetime
import json
import random
import re
import time
import traceback
import uuid
import typing as t
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
def _run_test_command(task_action: str, connection: ConnectionBase, command: str, expected: t.Optional[str]=None) -> None:
    """Runs the user specified test command until the host is able to run it properly"""
    display.vvvv(f'{task_action}: attempting post-reboot test command')
    rc, stdout, stderr = _execute_command(task_action, connection, command)
    if rc != 0:
        msg = f'{task_action}: Test command failed - rc: {rc}, stdout: {stdout}, stderr: {stderr}'
        raise _TestCommandFailure(msg)
    if expected and expected not in stdout:
        msg = f"{task_action}: Test command failed - '{expected}' was not in stdout: {stdout}"
        raise _TestCommandFailure(msg)