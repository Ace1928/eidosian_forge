from openstackclient.image.v2 import metadef_objects
from openstackclient.tests.unit.image.v2 import fakes
class TestMetadefObjectSet(fakes.TestImagev2):
    _metadef_namespace = fakes.create_one_metadef_namespace()
    _metadef_objects = fakes.create_one_metadef_object()
    new_metadef_object = fakes.create_one_metadef_object(attrs={'name': 'new_object_name'})

    def setUp(self):
        super().setUp()
        self.image_client.update_metadef_object.return_value = None
        self.cmd = metadef_objects.SetMetadefObject(self.app, None)

    def test_object_set_no_options(self):
        arglist = [self._metadef_namespace.namespace, self._metadef_objects.name]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)

    def test_object_set(self):
        arglist = [self._metadef_namespace.namespace, self._metadef_objects.name, '--name', 'new_object_name']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)