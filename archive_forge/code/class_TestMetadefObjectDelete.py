from openstackclient.image.v2 import metadef_objects
from openstackclient.tests.unit.image.v2 import fakes
class TestMetadefObjectDelete(fakes.TestImagev2):
    _metadef_namespace = fakes.create_one_metadef_namespace()
    _metadef_objects = fakes.create_one_metadef_object()

    def setUp(self):
        super().setUp()
        self.image_client.delete_metadef_object.return_value = None
        self.cmd = metadef_objects.DeleteMetadefObject(self.app, None)

    def test_object_delete(self):
        arglist = [self._metadef_namespace.namespace, self._metadef_objects.name]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)