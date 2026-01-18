import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
class TestField(object):

    def test_coerce_good_values(self):
        for in_val, out_val in self.coerce_good_values:
            self.assertEqual(out_val, self.field.coerce('obj', 'attr', in_val))

    def test_coerce_bad_values(self):
        for in_val in self.coerce_bad_values:
            self.assertRaises((TypeError, ValueError), self.field.coerce, 'obj', 'attr', in_val)

    def test_to_primitive(self):
        for in_val, prim_val in self.to_primitive_values:
            self.assertEqual(prim_val, self.field.to_primitive('obj', 'attr', in_val))

    def test_to_primitive_json_serializable(self):
        for in_val, _ in self.to_primitive_values:
            prim = self.field.to_primitive('obj', 'attr', in_val)
            jsencoded = jsonutils.dumps(prim)
            self.assertEqual(prim, jsonutils.loads(jsencoded))

    def test_from_primitive(self):

        class ObjectLikeThing(object):
            _context = 'context'
        for prim_val, out_val in self.from_primitive_values:
            from_prim = self.field.from_primitive(ObjectLikeThing, 'attr', prim_val)
            self.assertEqual(out_val, from_prim)
            self.field.coerce('obj', 'attr', from_prim)

    @abc.abstractmethod
    def test_stringify(self):
        """This test should validate stringify() format for new field types."""