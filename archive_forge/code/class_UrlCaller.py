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
class UrlCaller(object):

    def __init__(self):
        self._send_time = 0.5
        self._chunks = 25

    def send(self, url, data, status_cb=None):
        sleep_time = float(self._send_time) / self._chunks
        for i in range(0, len(data)):
            time.sleep(sleep_time)
            if status_cb:
                status_cb(float(i) / len(data))