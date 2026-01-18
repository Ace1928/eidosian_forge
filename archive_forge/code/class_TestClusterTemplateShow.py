import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
class TestClusterTemplateShow(TestClusterTemplate):
    attr = dict()
    attr['name'] = 'fake-ct-1'
    _cluster_template = magnum_fakes.FakeClusterTemplate.create_one_cluster_template(attr)
    columns = osc_ct.CLUSTER_TEMPLATE_ATTRIBUTES

    def setUp(self):
        super(TestClusterTemplateShow, self).setUp()
        datalist = map(lambda x: getattr(self._cluster_template, x), self.columns)
        self.show_data_list = tuple(map(lambda x: x if x is not None else '-', datalist))
        self.cluster_templates_mock.get = mock.Mock()
        self.cluster_templates_mock.get.return_value = self._cluster_template
        self.cmd = osc_ct.ShowClusterTemplate(self.app, None)

    def test_cluster_template_show(self):
        arglist = ['fake-ct-1']
        verifylist = [('cluster-template', 'fake-ct-1')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cluster_templates_mock.get.assert_called_with('fake-ct-1')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.show_data_list, data)

    def test_cluster_template_show_no_ct_fail(self):
        arglist = []
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)