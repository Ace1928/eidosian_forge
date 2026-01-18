from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import data_sources as api_ds
from saharaclient.osc.v1 import data_sources as osc_ds
from saharaclient.tests.unit.osc.v1 import test_data_sources as tds_v1
class TestCreateDataSource(TestDataSources):

    def setUp(self):
        super(TestCreateDataSource, self).setUp()
        self.ds_mock.create.return_value = api_ds.DataSources(None, DS_INFO)
        self.cmd = osc_ds.CreateDataSource(self.app, None)

    def test_data_sources_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_data_sources_create_required_options(self):
        arglist = ['source', '--type', 'swift', '--url', 'swift://container.sahara/object']
        verifylist = [('name', 'source'), ('type', 'swift'), ('url', 'swift://container.sahara/object')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        called_args = {'credential_pass': None, 'credential_user': None, 'data_source_type': 'swift', 'name': 'source', 'description': '', 'url': 'swift://container.sahara/object', 'is_public': False, 'is_protected': False, 's3_credentials': None}
        self.ds_mock.create.assert_called_once_with(**called_args)
        expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Name', 'Type', 'Url')
        self.assertEqual(expected_columns, columns)
        expected_data = ('Data Source for tests', 'id', True, True, 'source', 'swift', 'swift://container.sahara/object')
        self.assertEqual(expected_data, data)

    def test_data_sources_create_all_options(self):
        arglist = ['source', '--type', 'swift', '--url', 'swift://container.sahara/object', '--username', 'user', '--password', 'pass', '--description', 'Data Source for tests', '--public', '--protected']
        verifylist = [('name', 'source'), ('type', 'swift'), ('url', 'swift://container.sahara/object'), ('username', 'user'), ('password', 'pass'), ('description', 'Data Source for tests'), ('public', True), ('protected', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        called_args = {'credential_pass': 'pass', 'credential_user': 'user', 'data_source_type': 'swift', 'name': 'source', 'description': 'Data Source for tests', 'url': 'swift://container.sahara/object', 'is_protected': True, 'is_public': True, 's3_credentials': None}
        self.ds_mock.create.assert_called_once_with(**called_args)
        expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Name', 'Type', 'Url')
        self.assertEqual(expected_columns, columns)
        expected_data = ('Data Source for tests', 'id', True, True, 'source', 'swift', 'swift://container.sahara/object')
        self.assertEqual(expected_data, data)