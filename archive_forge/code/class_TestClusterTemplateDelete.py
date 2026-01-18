import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
class TestClusterTemplateDelete(TestClusterTemplate):

    def setUp(self):
        super(TestClusterTemplateDelete, self).setUp()
        self.cluster_templates_mock.delete = mock.Mock()
        self.cluster_templates_mock.delete.return_value = None
        self.cmd = osc_ct.DeleteClusterTemplate(self.app, None)

    def test_cluster_template_delete_one(self):
        arglist = ['foo']
        verifylist = [('cluster-templates', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cluster_templates_mock.delete.assert_called_with('foo')

    def test_cluster_template_delete_multiple(self):
        arglist = ['foo', 'bar']
        verifylist = [('cluster-templates', ['foo', 'bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cluster_templates_mock.delete.assert_has_calls([call('foo'), call('bar')])

    def test_cluster_template_delete_bad_uuid(self):
        self.cluster_templates_mock.delete.side_effect = osc_exceptions.NotFound(404)
        arglist = ['foo']
        verifylist = [('cluster-templates', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        returns = self.cmd.take_action(parsed_args)
        self.assertEqual(returns, None)

    def test_cluster_template_delete_no_uuid(self):
        arglist = []
        verifylist = [('cluster-templates', [])]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)