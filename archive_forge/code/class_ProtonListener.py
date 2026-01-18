import collections
import logging
import os
import threading
import uuid
import warnings
from debtcollector import removals
from oslo_config import cfg
from oslo_messaging.target import Target
from oslo_serialization import jsonutils
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_messaging._drivers.amqp1_driver.eventloop import compute_timeout
from oslo_messaging._drivers.amqp1_driver import opts
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common
@removals.removed_class('ProtonListener')
class ProtonListener(base.PollStyleListener):

    def __init__(self, driver):
        super(ProtonListener, self).__init__(driver.prefetch_size)
        self.driver = driver
        self.incoming = Queue()
        self.id = uuid.uuid4().hex

    def stop(self):
        self.incoming.stop()

    @base.batch_poll_helper
    def poll(self, timeout=None):
        qentry = self.incoming.pop(timeout)
        if qentry is None:
            return None
        return ProtonIncomingMessage(self, qentry['message'], qentry['disposition'])