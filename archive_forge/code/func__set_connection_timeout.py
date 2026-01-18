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
def _set_connection_timeout(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], timeout: float) -> None:
    """Sets the connection plugin connection_timeout option and resets the connection"""
    try:
        current_connection_timeout = connection.get_option('connection_timeout')
    except KeyError:
        return
    if timeout == current_connection_timeout:
        return
    display.vvvv(f'{task_action}: setting connect_timeout {timeout}')
    connection.set_option('connection_timeout', timeout)
    _reset_connection(task_action, connection, host_context, ignore_errors=True)