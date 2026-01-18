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
class RpcCall(object):

    def __init__(self, client, method, context):
        self.client = client
        self.method = method
        self.context = context

    def __call__(self, **kwargs):
        self.context['time'] = time.ctime()
        self.context['cast'] = False
        result = self.client.call(self.context, self.method, **kwargs)
        return result