import json
import logging
import os
import sys
import time
from oslo_utils import uuidutils
from taskflow import engines
from taskflow.listeners import printing
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.utils import misc
class ActivateDriver(task.Task):

    def __init__(self, resources):
        super(ActivateDriver, self).__init__(provides='sent_to')
        self._resources = resources
        self._url = 'http://blahblah.com'

    def execute(self, parsed_request):
        print('Sending billing data to %s' % self._url)
        url_sender = self._resources.url_handle
        url_sender.send(self._url, json.dumps(parsed_request), status_cb=self.update_progress)
        return self._url

    def update_progress(self, progress, **kwargs):
        super(ActivateDriver, self).update_progress(progress, **kwargs)
        print('%s is %0.2f%% done' % (self.name, progress * 100))