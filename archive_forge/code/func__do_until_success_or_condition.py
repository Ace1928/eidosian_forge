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
def _do_until_success_or_condition(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], action_desc: str, condition: t.Callable[[int], bool], func: t.Callable[..., T], *args: t.Any, **kwargs: t.Any) -> t.Optional[T]:
    """Runs the function multiple times ignoring errors until the condition is false"""
    fail_count = 0
    max_fail_sleep = 12
    reset_required = False
    last_error = None
    while fail_count == 0 or condition(fail_count):
        try:
            if reset_required:
                _reset_connection(task_action, connection, host_context)
                reset_required = False
            else:
                res = func(*args, **kwargs)
                display.vvvvv('%s: %s success' % (task_action, action_desc))
                return res
        except Exception as e:
            last_error = e
            if not isinstance(e, _TestCommandFailure):
                reset_required = True
            random_int = random.randint(0, 1000) / 1000
            fail_sleep = 2 ** fail_count + random_int
            if fail_sleep > max_fail_sleep:
                fail_sleep = max_fail_sleep + random_int
            try:
                error = str(e).splitlines()[-1]
            except IndexError:
                error = str(e)
            display.vvvvv("{action}: {desc} fail {e_type} '{err}', retrying in {sleep:.4} seconds...\n{tcb}".format(action=task_action, desc=action_desc, e_type=type(e).__name__, err=error, sleep=fail_sleep, tcb=traceback.format_exc()))
            fail_count += 1
            time.sleep(fail_sleep)
    if last_error:
        raise last_error
    return None