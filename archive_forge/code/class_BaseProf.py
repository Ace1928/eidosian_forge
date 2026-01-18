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
class BaseProf(with_metaclass(ABCMeta, threading.Thread)):

    def __init__(self, path, obj=None, writer=None):
        threading.Thread.__init__(self)
        self.obj = obj
        self.path = path
        self.max = 0
        self.running = True
        self.writer = writer

    def run(self):
        while self.running:
            self.poll()

    @abstractmethod
    def poll(self):
        pass