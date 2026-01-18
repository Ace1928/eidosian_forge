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
def _perform_reboot(task_action: str, connection: ConnectionBase, reboot_command: str, handle_abort: bool=True) -> None:
    """Runs the reboot command"""
    display.vv(f'{task_action}: rebooting server...')
    stdout = stderr = None
    try:
        rc, stdout, stderr = _execute_command(task_action, connection, reboot_command)
    except AnsibleConnectionFailure as e:
        display.vvvv(f'{task_action}: AnsibleConnectionFailure caught and handled: {e}')
        rc = 0
    if stdout:
        try:
            reboot_result = json.loads(stdout)
        except getattr(json.decoder, 'JSONDecodeError', ValueError):
            pass
        else:
            stdout = reboot_result.get('stdout', stdout)
            stderr = reboot_result.get('stderr', stderr)
            rc = int(reboot_result.get('rc', rc))
    if handle_abort and (rc == 1190 or (rc != 0 and stderr and ('(1190)' in stderr))):
        display.warning('A scheduled reboot was pre-empted by Ansible.')
        rc, stdout, stderr = _execute_command(task_action, connection, 'shutdown.exe /a')
        display.vvvv(f'{task_action}: result from trying to abort existing shutdown - rc: {rc}, stdout: {stdout}, stderr: {stderr}')
        return _perform_reboot(task_action, connection, reboot_command, handle_abort=False)
    if rc != 0:
        msg = f'{task_action}: Reboot command failed'
        raise _ReturnResultException(msg, rc=rc, stdout=stdout, stderr=stderr)