from __future__ import (absolute_import, division, print_function)
import hashlib
import hmac
import base64
import json
import uuid
import socket
import getpass
from datetime import datetime
from os.path import basename
from ansible.module_utils.urls import open_url
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.plugins.callback import CallbackBase
def _seconds_since_start(self, result):
    return (datetime.utcnow() - self.start_datetimes[result._task._uuid]).total_seconds()