import io
import tempfile
import textwrap
import six
from six.moves import configparser
import sys
from pbr.tests import base
from pbr import util
class TestMapFieldsParsingScenarios(base.BaseTestCase):
    scenarios = [('simple_project_urls', {'config_text': '\n                [metadata]\n                project_urls =\n                    Bug Tracker = https://bugs.launchpad.net/pbr/\n                    Documentation = https://docs.openstack.org/pbr/\n                    Source Code = https://opendev.org/openstack/pbr\n                ', 'expected_project_urls': {'Bug Tracker': 'https://bugs.launchpad.net/pbr/', 'Documentation': 'https://docs.openstack.org/pbr/', 'Source Code': 'https://opendev.org/openstack/pbr'}}), ('query_parameters', {'config_text': '\n                [metadata]\n                project_urls =\n                    Bug Tracker = https://bugs.launchpad.net/pbr/?query=true\n                    Documentation = https://docs.openstack.org/pbr/?foo=bar\n                    Source Code = https://git.openstack.org/cgit/openstack-dev/pbr/commit/?id=hash\n                ', 'expected_project_urls': {'Bug Tracker': 'https://bugs.launchpad.net/pbr/?query=true', 'Documentation': 'https://docs.openstack.org/pbr/?foo=bar', 'Source Code': 'https://git.openstack.org/cgit/openstack-dev/pbr/commit/?id=hash'}})]

    def test_project_url_parsing(self):
        config = config_from_ini(self.config_text)
        kwargs = util.setup_cfg_to_setup_kwargs(config)
        self.assertEqual(self.expected_project_urls, kwargs['project_urls'])