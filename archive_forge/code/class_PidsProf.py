from __future__ import (absolute_import, division, print_function)
import csv
import datetime
import os
import time
import threading
from abc import ABCMeta, abstractmethod
from functools import partial
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible.parsing.ajson import AnsibleJSONEncoder, json
from ansible.plugins.callback import CallbackBase
class PidsProf(BaseProf):

    def __init__(self, path, poll_interval=0.25, obj=None, writer=None):
        super(PidsProf, self).__init__(path, obj=obj, writer=writer)
        self._poll_interval = poll_interval

    def poll(self):
        with open(self.path) as f:
            val = int(f.read().strip())
        if val > self.max:
            self.max = val
        if self.writer:
            try:
                self.writer(time.time(), self.obj.get_name(), self.obj._uuid, val)
            except ValueError:
                self.running = False
        time.sleep(self._poll_interval)