import os
import tempfile
import textwrap
from openstack.config import loader
from openstack import exceptions
from openstack.tests.unit.config import base
class TestFixArgv(base.TestCase):

    def test_no_changes(self):
        argv = ['-a', '-b', '--long-arg', '--multi-value', 'key1=value1', '--multi-value', 'key2=value2']
        expected = argv[:]
        loader._fix_argv(argv)
        self.assertEqual(expected, argv)

    def test_replace(self):
        argv = ['-a', '-b', '--long-arg', '--multi_value', 'key1=value1', '--multi_value', 'key2=value2']
        expected = ['-a', '-b', '--long-arg', '--multi-value', 'key1=value1', '--multi-value', 'key2=value2']
        loader._fix_argv(argv)
        self.assertEqual(expected, argv)

    def test_mix(self):
        argv = ['-a', '-b', '--long-arg', '--multi_value', 'key1=value1', '--multi-value', 'key2=value2']
        self.assertRaises(exceptions.ConfigException, loader._fix_argv, argv)