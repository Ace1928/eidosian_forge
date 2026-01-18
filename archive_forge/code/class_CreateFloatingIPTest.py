import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import floatingips
class CreateFloatingIPTest(tests.TestCase):

    def setUp(self):
        super(CreateFloatingIPTest, self).setUp()
        self.create_floatingip = floatingips.CreateFloatingIP(shell.BlazarShell(), mock.Mock())

    def test_args2body(self):
        args = argparse.Namespace(network_id='1e17587e-a7ed-4b82-a17b-4beb32523e28', floating_ip_address='172.24.4.101')
        expected = {'network_id': '1e17587e-a7ed-4b82-a17b-4beb32523e28', 'floating_ip_address': '172.24.4.101'}
        ret = self.create_floatingip.args2body(args)
        self.assertDictEqual(ret, expected)