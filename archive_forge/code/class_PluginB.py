import uuid
from testtools import matchers
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class PluginB(object):

    def __init__(self, b):
        self.val = b