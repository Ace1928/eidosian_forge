from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
class TestCaseWithExampleMaps(TestCaseWithStore):

    def get_chk_bytes(self):
        if getattr(self, '_chk_bytes', None) is None:
            self._chk_bytes = super().get_chk_bytes()
        return self._chk_bytes

    def get_map(self, a_dict, maximum_size=100, search_key_func=None):
        c_map = self._get_map(a_dict, maximum_size=maximum_size, chk_bytes=self.get_chk_bytes(), search_key_func=search_key_func)
        return c_map

    def make_root_only_map(self, search_key_func=None):
        return self.get_map({(b'aaa',): b'initial aaa content', (b'abb',): b'initial abb content'}, search_key_func=search_key_func)

    def make_root_only_aaa_ddd_map(self, search_key_func=None):
        return self.get_map({(b'aaa',): b'initial aaa content', (b'ddd',): b'initial ddd content'}, search_key_func=search_key_func)

    def make_one_deep_map(self, search_key_func=None):
        return self.get_map({(b'aaa',): b'initial aaa content', (b'abb',): b'initial abb content', (b'ccc',): b'initial ccc content', (b'ddd',): b'initial ddd content'}, search_key_func=search_key_func)

    def make_two_deep_map(self, search_key_func=None):
        return self.get_map({(b'aaa',): b'initial aaa content', (b'abb',): b'initial abb content', (b'acc',): b'initial acc content', (b'ace',): b'initial ace content', (b'add',): b'initial add content', (b'adh',): b'initial adh content', (b'adl',): b'initial adl content', (b'ccc',): b'initial ccc content', (b'ddd',): b'initial ddd content'}, search_key_func=search_key_func)

    def make_one_deep_two_prefix_map(self, search_key_func=None):
        """Create a map with one internal node, but references are extra long.

        Otherwise has similar content to make_two_deep_map.
        """
        return self.get_map({(b'aaa',): b'initial aaa content', (b'add',): b'initial add content', (b'adh',): b'initial adh content', (b'adl',): b'initial adl content'}, search_key_func=search_key_func)

    def make_one_deep_one_prefix_map(self, search_key_func=None):
        """Create a map with one internal node, but references are extra long.

        Similar to make_one_deep_two_prefix_map, except the split is at the
        first char, rather than the second.
        """
        return self.get_map({(b'add',): b'initial add content', (b'adh',): b'initial adh content', (b'adl',): b'initial adl content', (b'bbb',): b'initial bbb content'}, search_key_func=search_key_func)