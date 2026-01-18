import os
import signal
import time
import fixtures
from pifpaf.drivers import rabbitmq
from oslo_messaging.tests.functional import utils
from oslo_messaging.tests import utils as test_utils
class ConnectedPortMatcher(object):

    def __init__(self, port):
        self.port = port

    def __eq__(self, data):
        return data.get('port') == self.port

    def __repr__(self):
        return '<ConnectedPortMatcher port=%d>' % self.port