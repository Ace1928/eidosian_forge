import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.compute import providers
from libcloud.utils.py3 import httplib
from libcloud.compute.base import NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.test.secrets import KAMATERA_PARAMS
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kamatera import KamateraNodeDriver
class KamateraMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('kamatera')

    def _service_server(self, method, url, body, headers):
        client_id, secret = (headers['AuthClientId'], headers['AuthSecret'])
        if client_id == 'nosuchuser' and secret == 'nopwd':
            body = self.fixtures.load('failed_auth.json')
            status = httplib.UNAUTHORIZED
        else:
            if url == '/service/server' and json.loads(body).get('ssh-key'):
                body = self.fixtures.load('create_server_sshkey.json')
            else:
                body = self.fixtures.load({'/service/server?datacenter=1': 'datacenters.json', '/service/server?sizes=1&datacenter=EU': 'sizes_datacenter_EU.json', '/service/server?images=1&datacenter=EU': 'images_datacenter_EU.json', '/service/server?capabilities=1&datacenter=EU': 'capabilities_datacenter_EU.json', '/service/server': 'create_server.json'}[url])
            status = httplib.OK
        return (status, body, {}, httplib.responses[status])

    def _service_queue(self, method, url, body, headers):
        if not hasattr(self, '_service_queue_call_count'):
            self._service_queue_call_count = 0
        self._service_queue_call_count += 1
        body = self.fixtures.load({'/service/queue?id=12345': 'queue_12345-%s.json' % self._service_queue_call_count}[url])
        status = httplib.OK
        return (status, body, {}, httplib.responses[status])

    def _service_server_info(self, method, url, body, headers):
        body = self.fixtures.load({'/service/server/info': 'server_info.json'}[url])
        status = httplib.OK
        return (status, body, {}, httplib.responses[status])

    def _service_servers(self, method, url, body, headers):
        body = self.fixtures.load({'/service/servers': 'servers.json'}[url])
        status = httplib.OK
        return (status, body, {}, httplib.responses[status])

    def _service_server_reboot(self, method, url, body, headers):
        body = self.fixtures.load({'/service/server/reboot': 'server_operation.json'}[url])
        status = httplib.OK
        return (status, body, {}, httplib.responses[status])