from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import metadef_properties
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestMetadefPropertyList(image_fakes.TestImagev2):
    _metadef_property = [image_fakes.create_one_metadef_property()]
    columns = ['name', 'title', 'type']

    def setUp(self):
        super().setUp()
        self.image_client.metadef_properties.side_effect = [self._metadef_property, []]
        self.cmd = metadef_properties.ListMetadefProperties(self.app, None)
        self.datalist = self._metadef_property

    def test_metadef_property_list(self):
        arglist = ['my-namespace']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(getattr(self.datalist[0], 'name'), next(data)[0])