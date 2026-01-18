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
class AzureLogAnalyticsSource(object):

    def __init__(self):
        self.ansible_check_mode = False
        self.ansible_playbook = ''
        self.ansible_version = ''
        self.session = str(uuid.uuid4())
        self.host = socket.gethostname()
        self.user = getpass.getuser()
        self.extra_vars = ''

    def __build_signature(self, date, workspace_id, shared_key, content_length):
        sigs = 'POST\n{0}\napplication/json\nx-ms-date:{1}\n/api/logs'.format(str(content_length), date)
        utf8_sigs = sigs.encode('utf-8')
        decoded_shared_key = base64.b64decode(shared_key)
        hmac_sha256_sigs = hmac.new(decoded_shared_key, utf8_sigs, digestmod=hashlib.sha256).digest()
        encoded_hash = base64.b64encode(hmac_sha256_sigs).decode('utf-8')
        signature = 'SharedKey {0}:{1}'.format(workspace_id, encoded_hash)
        return signature

    def __build_workspace_url(self, workspace_id):
        return 'https://{0}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01'.format(workspace_id)

    def __rfc1123date(self):
        return datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')

    def send_event(self, workspace_id, shared_key, state, result, runtime):
        if result._task_fields['args'].get('_ansible_check_mode') is True:
            self.ansible_check_mode = True
        if result._task_fields['args'].get('_ansible_version'):
            self.ansible_version = result._task_fields['args'].get('_ansible_version')
        if result._task._role:
            ansible_role = str(result._task._role)
        else:
            ansible_role = None
        data = {}
        data['uuid'] = result._task._uuid
        data['session'] = self.session
        data['status'] = state
        data['timestamp'] = self.__rfc1123date()
        data['host'] = self.host
        data['user'] = self.user
        data['runtime'] = runtime
        data['ansible_version'] = self.ansible_version
        data['ansible_check_mode'] = self.ansible_check_mode
        data['ansible_host'] = result._host.name
        data['ansible_playbook'] = self.ansible_playbook
        data['ansible_role'] = ansible_role
        data['ansible_task'] = result._task_fields
        if 'args' in data['ansible_task']:
            data['ansible_task'].pop('args')
        data['ansible_result'] = result._result
        if 'content' in data['ansible_result']:
            data['ansible_result'].pop('content')
        data['extra_vars'] = self.extra_vars
        jsondata = json.dumps({'event': data}, cls=AnsibleJSONEncoder, sort_keys=True)
        content_length = len(jsondata)
        rfc1123date = self.__rfc1123date()
        signature = self.__build_signature(rfc1123date, workspace_id, shared_key, content_length)
        workspace_url = self.__build_workspace_url(workspace_id)
        open_url(workspace_url, jsondata, headers={'content-type': 'application/json', 'Authorization': signature, 'Log-Type': 'ansible_playbook', 'x-ms-date': rfc1123date}, method='POST')