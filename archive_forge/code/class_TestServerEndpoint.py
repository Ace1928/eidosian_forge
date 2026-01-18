import os
import queue
import time
import uuid
import fixtures
from oslo_config import cfg
import oslo_messaging
from oslo_messaging._drivers.kafka_driver import kafka_options
from oslo_messaging.notify import notifier
from oslo_messaging.tests import utils as test_utils
class TestServerEndpoint(object):
    """This MessagingServer that will be used during functional testing."""

    def __init__(self):
        self.ival = 0
        self.sval = ''

    def add(self, ctxt, increment):
        self.ival += increment
        return self.ival

    def subtract(self, ctxt, increment):
        if self.ival < increment:
            raise ValueError("ival can't go negative!")
        self.ival -= increment
        return self.ival

    def append(self, ctxt, text):
        self.sval += text
        return self.sval

    def long_running_task(self, ctxt, seconds):
        time.sleep(seconds)