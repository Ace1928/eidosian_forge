from openstackclient.image.v2 import metadef_namespaces
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestMetadefNamespaceShow(image_fakes.TestImagev2):
    _metadef_namespace = image_fakes.create_one_metadef_namespace()
    expected_columns = ('created_at', 'display_name', 'namespace', 'owner', 'visibility')
    expected_data = (_metadef_namespace.created_at, _metadef_namespace.display_name, _metadef_namespace.namespace, _metadef_namespace.owner, _metadef_namespace.visibility)

    def setUp(self):
        super().setUp()
        self.image_client.get_metadef_namespace.return_value = self._metadef_namespace
        self.cmd = metadef_namespaces.ShowMetadefNamespace(self.app, None)

    def test_namespace_show_no_options(self):
        arglist = [self._metadef_namespace.namespace]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)