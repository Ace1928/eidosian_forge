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
def _wrap_conn_err(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except (AnsibleError, RequestException) as e:
        if ignore_errors:
            return False
        raise AnsibleError(e)
    return True