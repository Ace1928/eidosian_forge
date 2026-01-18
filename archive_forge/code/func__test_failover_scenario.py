import os
import signal
import time
import fixtures
from pifpaf.drivers import rabbitmq
from oslo_messaging.tests.functional import utils
from oslo_messaging.tests import utils as test_utils
def _test_failover_scenario(self, enable_cancel_on_failover=False):
    self.driver = os.environ.get('TRANSPORT_DRIVER')
    if self.driver not in self.DRIVERS:
        self.skipTest('TRANSPORT_DRIVER is not set to a rabbit driver')
    self.config(heartbeat_timeout_threshold=1, rpc_conn_pool_size=1, kombu_reconnect_delay=0, rabbit_retry_interval=0, rabbit_retry_backoff=0, enable_cancel_on_failover=enable_cancel_on_failover, group='oslo_messaging_rabbit')
    self.pifpaf = self.useFixture(rabbitmq.RabbitMQDriver(cluster=True, port=5692))
    self.url = self.pifpaf.env['PIFPAF_URL']
    self.n1 = self.pifpaf.env['PIFPAF_RABBITMQ_NODENAME1']
    self.n2 = self.pifpaf.env['PIFPAF_RABBITMQ_NODENAME2']
    self.n3 = self.pifpaf.env['PIFPAF_RABBITMQ_NODENAME3']
    self.pifpaf.stop_node(self.n2)
    self.pifpaf.stop_node(self.n3)
    self.servers = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.url, endpoint=self, names=['server']))
    self.useFixture(fixtures.MockPatch('oslo_messaging._drivers.impl_rabbit.random', side_effect=lambda x: x))
    self.client = self.servers.client(0)
    self.client.ping()
    self._check_ports(self.pifpaf.port)
    self.pifpaf.start_node(self.n2)
    self.assertEqual('callback done', self.client.kill_and_process())
    self.assertEqual('callback done', self.client.just_process())
    self._check_ports(self.pifpaf.get_port(self.n2))
    self.pifpaf.start_node(self.n3)
    time.sleep(0.1)
    self.pifpaf.kill_node(self.n2, signal=signal.SIGKILL)
    time.sleep(0.1)
    self.assertEqual('callback done', self.client.just_process())
    self._check_ports(self.pifpaf.get_port(self.n3))
    self.pifpaf.start_node(self.n1)
    time.sleep(0.1)
    self.pifpaf.kill_node(self.n3, signal=signal.SIGKILL)
    time.sleep(0.1)
    self.assertEqual('callback done', self.client.just_process())
    self._check_ports(self.pifpaf.get_port(self.n1))