import io
import sys
import textwrap
from unittest import mock
import fixtures
from oslotest import base
import tempfile
import testscenarios
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_config import generator
from oslo_config import types
import yaml
class HostAddressTestCase(base.BaseTestCase):
    opts = [cfg.HostAddressOpt('foo', help='foo option', default='0.0.0.0')]

    def test_host_address(self):
        config = [('namespace', [('alpha', self.opts)])]
        groups = generator._get_groups(config)
        out = io.StringIO()
        formatter = build_formatter(out)
        generator._output_opts(formatter, 'alpha', groups.pop('alpha'))
        result = out.getvalue()
        expected = textwrap.dedent('\n        [alpha]\n\n        #\n        # From namespace\n        #\n\n        # foo option (host address value)\n        #foo = 0.0.0.0\n        ').lstrip()
        self.assertEqual(expected, result)