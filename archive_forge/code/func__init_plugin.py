from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible import context
import socket
import uuid
import logging
from datetime import datetime
from ansible.plugins.callback import CallbackBase
def _init_plugin(self):
    if not self.disabled:
        self.logger = logging.getLogger('python-logstash-logger')
        self.logger.setLevel(logging.DEBUG)
        self.handler = logstash.TCPLogstashHandler(self.ls_server, self.ls_port, version=1, message_type=self.ls_type)
        self.logger.addHandler(self.handler)
        self.hostname = socket.gethostname()
        self.session = str(uuid.uuid4())
        self.errors = 0
        self.base_data = {'session': self.session, 'host': self.hostname}
        if self.ls_pre_command is not None:
            self.base_data['ansible_pre_command_output'] = os.popen(self.ls_pre_command).read()
        if context.CLIARGS is not None:
            self.base_data['ansible_checkmode'] = context.CLIARGS.get('check')
            self.base_data['ansible_tags'] = context.CLIARGS.get('tags')
            self.base_data['ansible_skip_tags'] = context.CLIARGS.get('skip_tags')
            self.base_data['inventory'] = context.CLIARGS.get('inventory')