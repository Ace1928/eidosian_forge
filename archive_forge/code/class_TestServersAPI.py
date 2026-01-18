import time
from novaclient.tests.functional import base
class TestServersAPI(base.ClientTestBase):

    def test_server_ips(self):
        server_name = 'test_server'
        initial_server = self.client.servers.create(server_name, self.image, self.flavor, nics=[{'net-id': self.network.id}])
        self.addCleanup(initial_server.delete)
        for x in range(60):
            server = self.client.servers.get(initial_server)
            if server.status == 'ACTIVE':
                break
            else:
                time.sleep(1)
        else:
            self.fail('Server %s did not go ACTIVE after 60s' % server)
        ips = self.client.servers.ips(server)
        self.assertIn(self.network.name, ips)