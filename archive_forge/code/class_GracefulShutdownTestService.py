import logging
import multiprocessing
import os
import signal
import socket
import time
import traceback
from unittest import mock
import eventlet
from eventlet import event
from oslotest import base as test_base
from oslo_service import service
from oslo_service.tests import base
from oslo_service.tests import eventlet_service
class GracefulShutdownTestService(service.Service):

    def __init__(self):
        super(GracefulShutdownTestService, self).__init__()
        self.finished_task = event.Event()

    def start(self, sleep_amount):

        def sleep_and_send(finish_event):
            time.sleep(sleep_amount)
            finish_event.send()
        self.tg.add_thread(sleep_and_send, self.finished_task)