import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
class CreateResourcesTest(utils.BaseTestCase):

    def setUp(self):
        super(CreateResourcesTest, self).setUp()
        self.client = mock.MagicMock()

    @mock.patch.object(create_resources, 'create_nodes', autospec=True)
    @mock.patch.object(create_resources, 'create_chassis', autospec=True)
    @mock.patch.object(jsonschema, 'validate', autospec=True)
    @mock.patch.object(create_resources, 'load_from_file', side_effect=[valid_json], autospec=True)
    def test_create_resources(self, mock_load, mock_validate, mock_chassis, mock_nodes):
        resources_files = ['file.json']
        create_resources.create_resources(self.client, resources_files)
        mock_load.assert_has_calls([mock.call('file.json')])
        mock_validate.assert_called_once_with(valid_json, mock.ANY)
        mock_chassis.assert_called_once_with(self.client, valid_json['chassis'])
        mock_nodes.assert_called_once_with(self.client, valid_json['nodes'])

    @mock.patch.object(create_resources, 'create_nodes', autospec=True)
    @mock.patch.object(create_resources, 'create_chassis', autospec=True)
    @mock.patch.object(jsonschema, 'validate', autospec=True)
    @mock.patch.object(create_resources, 'load_from_file', side_effect=exc.ClientException, autospec=True)
    def test_create_resources_cannot_read_schema(self, mock_load, mock_validate, mock_chassis, mock_nodes):
        resources_files = ['file.json']
        self.assertRaises(exc.ClientException, create_resources.create_resources, self.client, resources_files)
        mock_load.assert_called_once_with('file.json')
        self.assertFalse(mock_validate.called)
        self.assertFalse(mock_chassis.called)
        self.assertFalse(mock_nodes.called)

    @mock.patch.object(create_resources, 'create_nodes', autospec=True)
    @mock.patch.object(create_resources, 'create_chassis', autospec=True)
    @mock.patch.object(jsonschema, 'validate', side_effect=jsonschema.ValidationError(''), autospec=True)
    @mock.patch.object(create_resources, 'load_from_file', side_effect=[schema_pov_invalid_json], autospec=True)
    def test_create_resources_validation_fails(self, mock_load, mock_validate, mock_chassis, mock_nodes):
        resources_files = ['file.json']
        self.assertRaises(exc.ClientException, create_resources.create_resources, self.client, resources_files)
        mock_load.assert_has_calls([mock.call('file.json')])
        mock_validate.assert_called_once_with(schema_pov_invalid_json, mock.ANY)
        self.assertFalse(mock_chassis.called)
        self.assertFalse(mock_nodes.called)

    @mock.patch.object(create_resources, 'create_nodes', autospec=True)
    @mock.patch.object(create_resources, 'create_chassis', autospec=True)
    @mock.patch.object(jsonschema, 'validate', side_effect=[None, jsonschema.ValidationError('')], autospec=True)
    @mock.patch.object(create_resources, 'load_from_file', side_effect=[valid_json, schema_pov_invalid_json], autospec=True)
    def test_create_resources_validation_fails_multiple(self, mock_load, mock_validate, mock_chassis, mock_nodes):
        resources_files = ['file.json', 'file2.json']
        self.assertRaises(exc.ClientException, create_resources.create_resources, self.client, resources_files)
        mock_load.assert_has_calls([mock.call('file.json'), mock.call('file2.json')])
        mock_validate.assert_has_calls([mock.call(valid_json, mock.ANY), mock.call(schema_pov_invalid_json, mock.ANY)])
        self.assertFalse(mock_chassis.called)
        self.assertFalse(mock_nodes.called)

    @mock.patch.object(create_resources, 'create_nodes', autospec=True)
    @mock.patch.object(create_resources, 'create_chassis', autospec=True)
    @mock.patch.object(jsonschema, 'validate', autospec=True)
    @mock.patch.object(create_resources, 'load_from_file', side_effect=[ironic_pov_invalid_json], autospec=True)
    def test_create_resources_ironic_fails_to_create(self, mock_load, mock_validate, mock_chassis, mock_nodes):
        mock_nodes.return_value = [exc.ClientException('cannot create that')]
        mock_chassis.return_value = []
        resources_files = ['file.json']
        self.assertRaises(exc.ClientException, create_resources.create_resources, self.client, resources_files)
        mock_load.assert_has_calls([mock.call('file.json')])
        mock_validate.assert_called_once_with(ironic_pov_invalid_json, mock.ANY)
        mock_chassis.assert_called_once_with(self.client, [])
        mock_nodes.assert_called_once_with(self.client, ironic_pov_invalid_json['nodes'])