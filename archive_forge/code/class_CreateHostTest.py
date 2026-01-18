import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import hosts
class CreateHostTest(tests.TestCase):

    def setUp(self):
        super(CreateHostTest, self).setUp()
        self.create_host = hosts.CreateHost(shell.BlazarShell(), mock.Mock())

    def test_args2body(self):
        args = argparse.Namespace(name='test-host', extra_capabilities=['extra_key1=extra_value1', 'extra_key2=extra_value2'])
        expected = {'name': 'test-host', 'extra_key1': 'extra_value1', 'extra_key2': 'extra_value2'}
        ret = self.create_host.args2body(args)
        self.assertDictEqual(ret, expected)