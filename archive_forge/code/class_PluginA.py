import uuid
from testtools import matchers
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class PluginA(object):

    def __init__(self, a):
        self.val = a