import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.utils.docker import HubClient
class DockerMockHttp(MockHttp):
    fixtures = ContainerFileFixtures('docker_utils')

    def _v2_repositories_library_ubuntu_tags_latest(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v2_repositories_library_ubuntu_tags_latest.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_repositories_library_ubuntu_tags(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v2_repositories_library_ubuntu_tags.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_repositories_library_ubuntu(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v2_repositories_library_ubuntu.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])