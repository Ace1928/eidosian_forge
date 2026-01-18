from bisect import bisect_right
from .die import DIE
from ..common.utils import dwarf_assert
def _get_cached_DIE(self, offset):
    """ Given a DIE offset, look it up in the cache.  If not present,
            parse the DIE and insert it into the cache.

            offset:
                The offset of the DIE in the debug_info section to retrieve.

            The stream reference is copied from the top DIE.  The top die will
            also be parsed and cached if needed.

            See also get_DIE_from_refaddr(self, refaddr).
        """
    top_die_stream = self.get_top_DIE().stream
    i = bisect_right(self._diemap, offset)
    if offset == self._diemap[i - 1]:
        die = self._dielist[i - 1]
    else:
        die = DIE(cu=self, stream=top_die_stream, offset=offset)
        self._dielist.insert(i, die)
        self._diemap.insert(i, offset)
    return die