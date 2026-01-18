from .. import errors as errors
from .. import identitymap as identitymap
from . import TestCase
class TestIdentityMap(TestCase):

    def test_symbols(self):
        from ..identitymap import IdentityMap

    def test_construct(self):
        identitymap.IdentityMap()

    def test_add_weave(self):
        map = identitymap.IdentityMap()
        weave = 'foo'
        map.add_weave('id', weave)
        self.assertEqual(weave, map.find_weave('id'))

    def test_double_add_weave(self):
        map = identitymap.IdentityMap()
        weave = 'foo'
        map.add_weave('id', weave)
        self.assertRaises(errors.BzrError, map.add_weave, 'id', weave)
        self.assertEqual(weave, map.find_weave('id'))

    def test_remove_object(self):
        map = identitymap.IdentityMap()
        weave = 'foo'
        map.add_weave('id', weave)
        map.remove_object(weave)
        map.add_weave('id', weave)