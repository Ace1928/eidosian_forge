import argparse
from unittest import mock
import testtools
from ironicclient.osc import plugin
from ironicclient.tests.unit.osc import fakes
from ironicclient.v1 import client
@mock.patch.object(plugin, 'OS_BAREMETAL_API_LATEST', new=True)
@mock.patch.object(argparse.ArgumentParser, 'add_argument', autospec=True)
class BuildOptionParserTest(testtools.TestCase):

    @mock.patch.object(plugin.utils, 'env', lambda x: None)
    def test_build_option_parser(self, mock_add_argument):
        parser = argparse.ArgumentParser()
        mock_add_argument.reset_mock()
        plugin.build_option_parser(parser)
        version_list = ['1'] + ['1.%d' % i for i in range(1, plugin.LAST_KNOWN_API_VERSION + 1)] + ['latest']
        mock_add_argument.assert_called_once_with(mock.ANY, '--os-baremetal-api-version', action=plugin.ReplaceLatestVersion, choices=version_list, default=plugin.LATEST_VERSION, help=mock.ANY, metavar='<baremetal-api-version>')
        self.assertTrue(plugin.OS_BAREMETAL_API_LATEST)

    @mock.patch.object(plugin.utils, 'env', lambda x: 'latest')
    def test_build_option_parser_env_latest(self, mock_add_argument):
        parser = argparse.ArgumentParser()
        mock_add_argument.reset_mock()
        plugin.build_option_parser(parser)
        version_list = ['1'] + ['1.%d' % i for i in range(1, plugin.LAST_KNOWN_API_VERSION + 1)] + ['latest']
        mock_add_argument.assert_called_once_with(mock.ANY, '--os-baremetal-api-version', action=plugin.ReplaceLatestVersion, choices=version_list, default=plugin.LATEST_VERSION, help=mock.ANY, metavar='<baremetal-api-version>')
        self.assertTrue(plugin.OS_BAREMETAL_API_LATEST)

    @mock.patch.object(plugin.utils, 'env', lambda x: '1.1')
    def test_build_option_parser_env(self, mock_add_argument):
        parser = argparse.ArgumentParser()
        mock_add_argument.reset_mock()
        plugin.build_option_parser(parser)
        version_list = ['1'] + ['1.%d' % i for i in range(1, plugin.LAST_KNOWN_API_VERSION + 1)] + ['latest']
        mock_add_argument.assert_called_once_with(mock.ANY, '--os-baremetal-api-version', action=plugin.ReplaceLatestVersion, choices=version_list, default='1.1', help=mock.ANY, metavar='<baremetal-api-version>')
        self.assertFalse(plugin.OS_BAREMETAL_API_LATEST)