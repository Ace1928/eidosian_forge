import os
import requests
import subprocess
import time
import uuid
import concurrent.futures
from oslo_config import cfg
from testtools import matchers
import oslo_messaging
from oslo_messaging.tests.functional import utils
class MetricsTestCase(utils.SkipIfNoTransportURL):

    def setUp(self):
        super(MetricsTestCase, self).setUp(conf=cfg.ConfigOpts())
        if self.rpc_url.startswith('kafka://'):
            self.skipTest('kafka does not support RPC API')
        self.config(metrics_enabled=True, group='oslo_messaging_metrics')

    def test_functional(self):
        self.config(metrics_socket_file='/var/tmp/metrics_collector.sock', group='oslo_messaging_metrics')
        metric_server = subprocess.Popen(['python3', '-m', 'oslo_metrics'])
        time.sleep(1)
        group = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url))
        client = group.client(1)
        client.add(increment=1)
        time.sleep(1)
        r = requests.get('http://localhost:3000', timeout=10)
        for line in r.text.split('\n'):
            if 'client_invocation_start_total{' in line:
                self.assertEqual('1.0', line[-3:])
            elif 'client_invocation_end_total{' in line:
                self.assertEqual('1.0', line[-3:])
            elif 'client_processing_seconds_count{' in line:
                self.assertEqual('1.0', line[-3:])
        metric_server.terminate()