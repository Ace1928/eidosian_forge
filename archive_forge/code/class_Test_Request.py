import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class Test_Request(base.TestCase):

    def test_create(self):
        uri = 1
        body = 2
        headers = 3
        sot = resource._Request(uri, body, headers)
        self.assertEqual(uri, sot.url)
        self.assertEqual(body, sot.body)
        self.assertEqual(headers, sot.headers)