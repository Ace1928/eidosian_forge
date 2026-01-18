from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
class QueueStatus(object):

    def __init__(self, module, current_status):
        self.module = module
        self.current_status = current_status
        self.desired_status = self.module.params.get('status')

    def run(self):
        if GcpRequest({'status': self.current_status}) == GcpRequest({'status': self.desired_status}):
            return
        elif self.desired_status == 'PAUSED':
            self.stop()
        elif self.desired_status == 'RUNNING':
            self.start()

    def start(self):
        auth = GcpSession(self.module, 'cloudtasks')
        return_if_object(self.module, auth.post(self._start_url()))

    def stop(self):
        auth = GcpSession(self.module, 'cloudtasks')
        return_if_object(self.module, auth.post(self._stop_url()))

    def _start_url(self):
        return 'https://cloudtasks.googleapis.com/v2/projects/{project}/locations/{location}/queues/{name}:resume'.format(**self.module.params)

    def _stop_url(self):
        return 'https://cloudtasks.googleapis.com/v2/projects/{project}/locations/{location}/queues/{name}:pause'.format(**self.module.params)