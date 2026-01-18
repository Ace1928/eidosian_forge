import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.utils.docker import HubClient
class DockerUtilitiesTestCase(unittest.TestCase):

    def setUp(self):
        HubClient.connectionCls.conn_class = DockerMockHttp
        DockerMockHttp.type = None
        DockerMockHttp.use_param = 'a'
        self.driver = HubClient()

    def test_list_tags(self):
        tags = self.driver.list_images('ubuntu', max_count=100)
        self.assertEqual(len(tags), 88)
        self.assertEqual(tags[0].name, 'registry.hub.docker.com/ubuntu:xenial')

    def test_get_repository(self):
        repo = self.driver.get_repository('ubuntu')
        self.assertEqual(repo['name'], 'ubuntu')

    def test_get_image(self):
        image = self.driver.get_image('ubuntu', 'latest')
        self.assertEqual(image.id, '2343')
        self.assertEqual(image.name, 'registry.hub.docker.com/ubuntu:latest')
        self.assertEqual(image.path, 'registry.hub.docker.com/ubuntu:latest')