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
def __build_workspace_url(self, workspace_id):
    return 'https://{0}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01'.format(workspace_id)