from .. import errors as errors
from .. import identitymap as identitymap
from . import TestCase
class TestNullIdentityMap(TestCase):

    def test_symbols(self):
        from ..identitymap import NullIdentityMap

    def test_construct(self):
        identitymap.NullIdentityMap()

    def test_add_weave(self):
        map = identitymap.NullIdentityMap()
        weave = 'foo'
        map.add_weave('id', weave)
        self.assertEqual(None, map.find_weave('id'))

    def test_double_add_weave(self):
        map = identitymap.NullIdentityMap()
        weave = 'foo'
        map.add_weave('id', weave)
        map.add_weave('id', weave)
        self.assertEqual(None, map.find_weave('id'))

    def test_null_identity_map_has_no_remove(self):
        map = identitymap.NullIdentityMap()
        self.assertEqual(None, getattr(map, 'remove_object', None))