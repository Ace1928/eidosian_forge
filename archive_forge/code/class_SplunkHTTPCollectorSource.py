from __future__ import (absolute_import, division, print_function)
import json
import uuid
import socket
import getpass
from datetime import datetime
from os.path import basename
from ansible.module_utils.urls import open_url
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.plugins.callback import CallbackBase
class SplunkHTTPCollectorSource(object):

    def __init__(self):
        self.ansible_check_mode = False
        self.ansible_playbook = ''
        self.ansible_version = ''
        self.session = str(uuid.uuid4())
        self.host = socket.gethostname()
        self.ip_address = socket.gethostbyname(socket.gethostname())
        self.user = getpass.getuser()

    def send_event(self, url, authtoken, validate_certs, include_milliseconds, batch, state, result, runtime):
        if result._task_fields['args'].get('_ansible_check_mode') is True:
            self.ansible_check_mode = True
        if result._task_fields['args'].get('_ansible_version'):
            self.ansible_version = result._task_fields['args'].get('_ansible_version')
        if result._task._role:
            ansible_role = str(result._task._role)
        else:
            ansible_role = None
        if 'args' in result._task_fields:
            del result._task_fields['args']
        data = {}
        data['uuid'] = result._task._uuid
        data['session'] = self.session
        if batch is not None:
            data['batch'] = batch
        data['status'] = state
        if include_milliseconds:
            time_format = '%Y-%m-%d %H:%M:%S.%f +0000'
        else:
            time_format = '%Y-%m-%d %H:%M:%S +0000'
        data['timestamp'] = datetime.utcnow().strftime(time_format)
        data['host'] = self.host
        data['ip_address'] = self.ip_address
        data['user'] = self.user
        data['runtime'] = runtime
        data['ansible_version'] = self.ansible_version
        data['ansible_check_mode'] = self.ansible_check_mode
        data['ansible_host'] = result._host.name
        data['ansible_playbook'] = self.ansible_playbook
        data['ansible_role'] = ansible_role
        data['ansible_task'] = result._task_fields
        data['ansible_result'] = result._result
        jsondata = json.dumps(data, cls=AnsibleJSONEncoder, sort_keys=True)
        jsondata = '{"event":' + jsondata + '}'
        open_url(url, jsondata, headers={'Content-type': 'application/json', 'Authorization': 'Splunk ' + authtoken}, method='POST', validate_certs=validate_certs)