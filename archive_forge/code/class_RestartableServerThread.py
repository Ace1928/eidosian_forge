import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class RestartableServerThread(object):

    def __init__(self, server):
        self.server = server
        self.thread = None

    def start(self):
        if self.thread is None:
            self.thread = test_utils.ServerThreadHelper(self.server)
            self.thread.start()

    def stop(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread.join(timeout=15)
            ret = self.thread.is_alive()
            self.thread = None
            return ret
        return True