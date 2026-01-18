import sys
import random
import string
import unittest
from libcloud.utils.py3 import httplib
from libcloud.common.gandi import GandiException
from libcloud.test.secrets import GANDI_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gandi import GandiNodeDriver
from libcloud.test.common.test_gandi import BaseGandiMockHttp
class GandiRatingTests(unittest.TestCase):
    """Tests where rating model is involved"""
    node_name = 'test2'

    def setUp(self):
        GandiNodeDriver.connectionCls.conn_class = GandiMockRatingHttp
        GandiMockRatingHttp.type = None
        self.driver = GandiNodeDriver(*GANDI_PARAMS)

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 4)

    def test_create_node(self):
        login = 'libcloud'
        passwd = ''.join((random.choice(string.ascii_letters) for i in range(10)))
        loc = list(filter(lambda x: 'france' in x.country.lower(), self.driver.list_locations()))[0]
        images = self.driver.list_images(loc)
        images = [x for x in images if x.name.lower().startswith('debian')]
        img = list(filter(lambda x: '5' in x.name, images))[0]
        size = self.driver.list_sizes()[0]
        node = self.driver.create_node(name=self.node_name, login=login, password=passwd, image=img, location=loc, size=size)
        self.assertEqual(node.name, self.node_name)