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
def __build_signature(self, date, workspace_id, shared_key, content_length):
    sigs = 'POST\n{0}\napplication/json\nx-ms-date:{1}\n/api/logs'.format(str(content_length), date)
    utf8_sigs = sigs.encode('utf-8')
    decoded_shared_key = base64.b64decode(shared_key)
    hmac_sha256_sigs = hmac.new(decoded_shared_key, utf8_sigs, digestmod=hashlib.sha256).digest()
    encoded_hash = base64.b64encode(hmac_sha256_sigs).decode('utf-8')
    signature = 'SharedKey {0}:{1}'.format(workspace_id, encoded_hash)
    return signature