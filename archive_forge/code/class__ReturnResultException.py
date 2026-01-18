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
class _ReturnResultException(Exception):
    """Used to sneak results back to the return dict from an exception"""

    def __init__(self, msg, **result):
        super().__init__(msg)
        self.result = result