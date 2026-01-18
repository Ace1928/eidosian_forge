import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
class BTreeTestCase(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.overrideAttr(btree_index, '_RESERVED_HEADER_BYTES', 100)

    def make_nodes(self, count, key_elements, reference_lists):
        """Generate count*key_elements sample nodes."""

        def _pos_to_key(pos, lead=b''):
            return (lead + b'%d' % pos * 40,)
        keys = []
        for prefix_pos in range(key_elements):
            if key_elements - 1:
                prefix = _pos_to_key(prefix_pos)
            else:
                prefix = ()
            for pos in range(count):
                key = prefix + _pos_to_key(pos)
                value = b'value:%d' % pos
                if reference_lists:
                    refs = []
                    for list_pos in range(reference_lists):
                        refs.append([])
                        for ref_pos in range(list_pos + pos % 2):
                            if pos % 2:
                                refs[-1].append(prefix + _pos_to_key(pos - 1, b'ref'))
                            else:
                                refs[-1].append(prefix + _pos_to_key(ref_pos, b'ref'))
                        refs[-1] = tuple(refs[-1])
                    refs = tuple(refs)
                else:
                    refs = ()
                keys.append((key, value, refs))
        return keys

    def shrink_page_size(self):
        """Shrink the default page size so that less fits in a page."""
        self.overrideAttr(btree_index, '_PAGE_SIZE')
        btree_index._PAGE_SIZE = 2048

    def assertEqualApproxCompressed(self, expected, actual, slop=6):
        """Check a count of compressed bytes is approximately as expected

        Relying on compressed length being stable even with fixed inputs is
        slightly bogus, but zlib is stable enough that this mostly works.
        """
        if not expected - slop < actual < expected + slop:
            self.fail('Expected around %d bytes compressed but got %d' % (expected, actual))