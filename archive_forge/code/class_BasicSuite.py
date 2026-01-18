import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
class BasicSuite(unittest.TestCase):

    def setUp(self):
        self.random_id = str(random.randint(1, 1000000))
        self.app_name = 'app-' + self.random_id
        self.app_version = 'version-' + self.random_id
        self.template = 'template-' + self.random_id
        self.environment = 'environment-' + self.random_id
        self.beanstalk = Layer1Wrapper()