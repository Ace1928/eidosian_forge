import functools
from unittest import mock
import uuid
from keystoneauth1 import loading
from keystoneauth1.loading import base
from keystoneauth1 import plugin
from keystoneauth1.tests.unit import utils
class MockManager(object):

    def __init__(self, driver):
        self.driver = driver