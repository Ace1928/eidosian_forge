from heat.common import identifier
from heat.tests import common
class ResourceIdentifierTest(common.HeatTestCase):

    def test_resource_init_no_path(self):
        si = identifier.HeatIdentifier('t', 's', 'i')
        ri = identifier.ResourceIdentifier(resource_name='r', **si)
        self.assertEqual('/resources/r', ri.path)

    def test_resource_init_path(self):
        si = identifier.HeatIdentifier('t', 's', 'i')
        pi = identifier.ResourceIdentifier(resource_name='p', **si)
        ri = identifier.ResourceIdentifier(resource_name='r', **pi)
        self.assertEqual('/resources/p/resources/r', ri.path)

    def test_resource_init_from_dict(self):
        hi = identifier.HeatIdentifier('t', 's', 'i', '/resources/r')
        ri = identifier.ResourceIdentifier(**hi)
        self.assertEqual(hi, ri)

    def test_resource_stack(self):
        si = identifier.HeatIdentifier('t', 's', 'i')
        ri = identifier.ResourceIdentifier(resource_name='r', **si)
        self.assertEqual(si, ri.stack())

    def test_resource_id(self):
        ri = identifier.ResourceIdentifier('t', 's', 'i', '', 'r')
        self.assertEqual('r', ri.resource_name)

    def test_resource_name_slash(self):
        self.assertRaises(ValueError, identifier.ResourceIdentifier, 't', 's', 'i', 'p', 'r/r')