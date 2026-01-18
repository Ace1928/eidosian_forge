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
def _ssm_retry(func):
    """
    Decorator to retry in the case of a connection failure
    Will retry if:
    * an exception is caught
    Will not retry if
    * remaining_tries is <2
    * retries limit reached
    """

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        remaining_tries = int(self.get_option('reconnection_retries')) + 1
        cmd_summary = f'{args[0]}...'
        for attempt in range(remaining_tries):
            try:
                return_tuple = func(self, *args, **kwargs)
                self._vvvv(f'ssm_retry: (success) {to_text(return_tuple)}')
                break
            except (AnsibleConnectionFailure, Exception) as e:
                if attempt == remaining_tries - 1:
                    raise
                pause = 2 ** attempt - 1
                pause = min(pause, 30)
                if isinstance(e, AnsibleConnectionFailure):
                    msg = f'ssm_retry: attempt: {attempt}, cmd ({cmd_summary}), pausing for {pause} seconds'
                else:
                    msg = f'ssm_retry: attempt: {attempt}, caught exception({e}) from cmd ({cmd_summary}), pausing for {pause} seconds'
                self._vv(msg)
                time.sleep(pause)
                self.close()
                continue
        return return_tuple
    return wrapped