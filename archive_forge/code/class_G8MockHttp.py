import sys
import json
import time
import base64
import unittest
from unittest import mock
import libcloud.common.gig_g8
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, StorageVolume
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gig_g8 import G8Network, G8NodeDriver, G8PortForward
class G8MockHttp(MockHttp):
    """Fixtures needed for tests related to rating model"""
    fixtures = ComputeFileFixtures('gig_g8')

    def __getattr__(self, key):

        def method(method, path, params, headers):
            response = self.fixtures.load('{}_{}.json'.format(method, key.lstrip('_')))
            return (httplib.OK, response, {}, httplib.responses[httplib.OK])
        return method