import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
class TestClusterTemplateList(TestClusterTemplate):
    attr = dict()
    attr['name'] = 'fake-ct-1'
    _cluster_template = magnum_fakes.FakeClusterTemplate.create_one_cluster_template(attr)
    attr['name'] = 'fake-ct-2'
    _cluster_template2 = magnum_fakes.FakeClusterTemplate.create_one_cluster_template(attr)
    columns = ['uuid', 'name', 'tags']
    datalist = ((_cluster_template.uuid, _cluster_template.name, _cluster_template.tags), (_cluster_template2.uuid, _cluster_template2.name, _cluster_template.tags))

    def setUp(self):
        super(TestClusterTemplateList, self).setUp()
        self.cluster_templates_mock.list = mock.Mock()
        self.cluster_templates_mock.list.return_value = [self._cluster_template, self._cluster_template2]
        self.cmd = osc_ct.ListTemplateCluster(self.app, None)

    def test_cluster_template_list_no_options(self):
        arglist = []
        verifylist = [('limit', None), ('sort_key', None), ('sort_dir', None), ('fields', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cluster_templates_mock.list.assert_called_with(limit=None, sort_dir=None, sort_key=None)
        self.assertEqual(self.columns, columns)
        index = 0
        for d in data:
            self.assertEqual(self.datalist[index], d)
            index += 1

    def test_cluster_template_list_options(self):
        arglist = ['--limit', '1', '--sort-key', 'key', '--sort-dir', 'asc', '--fields', 'field1,field2']
        verifylist = [('limit', 1), ('sort_key', 'key'), ('sort_dir', 'asc'), ('fields', 'field1,field2')]
        verifycolumns = self.columns + ['field1', 'field2']
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cluster_templates_mock.list.assert_called_with(limit=1, sort_dir='asc', sort_key='key')
        self.assertEqual(verifycolumns, columns)

    def test_cluster_template_list_bad_sort_dir_fail(self):
        arglist = ['--sort-dir', 'foo']
        verifylist = [('limit', None), ('sort_key', None), ('sort_dir', 'foo'), ('fields', None)]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)