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
def _do_until_success_or_retry_limit(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], action_desc: str, retries: int, func: t.Callable[..., T], *args: t.Any, **kwargs: t.Any) -> t.Optional[T]:
    """Runs the function multiple times ignoring errors until the retry limit is hit"""

    def wait_condition(idx):
        return idx < retries
    return _do_until_success_or_condition(task_action, connection, host_context, action_desc, wait_condition, func, *args, **kwargs)