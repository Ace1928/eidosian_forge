from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _fancy_replace(self, a, alo, ahi, b, blo, bhi):
    """
        When replacing one block of lines with another, search the blocks
        for *similar* lines; the best-matching pair (if any) is used as a
        synch point, and intraline difference marking is done on the
        similar pair. Lots of work, but often worth it.

        Example:

        >>> d = Differ()
        >>> results = d._fancy_replace(['abcDefghiJkl\\n'], 0, 1,
        ...                            ['abcdefGhijkl\\n'], 0, 1)
        >>> print(''.join(results), end="")
        - abcDefghiJkl
        ?    ^  ^  ^
        + abcdefGhijkl
        ?    ^  ^  ^
        """
    best_ratio, cutoff = (0.74, 0.75)
    cruncher = SequenceMatcher(self.charjunk)
    eqi, eqj = (None, None)
    for j in range(blo, bhi):
        bj = b[j]
        cruncher.set_seq2(bj)
        for i in range(alo, ahi):
            ai = a[i]
            if ai == bj:
                if eqi is None:
                    eqi, eqj = (i, j)
                continue
            cruncher.set_seq1(ai)
            if cruncher.real_quick_ratio() > best_ratio and cruncher.quick_ratio() > best_ratio and (cruncher.ratio() > best_ratio):
                best_ratio, best_i, best_j = (cruncher.ratio(), i, j)
    if best_ratio < cutoff:
        if eqi is None:
            yield from self._plain_replace(a, alo, ahi, b, blo, bhi)
            return
        best_i, best_j, best_ratio = (eqi, eqj, 1.0)
    else:
        eqi = None
    yield from self._fancy_helper(a, alo, best_i, b, blo, best_j)
    aelt, belt = (a[best_i], b[best_j])
    if eqi is None:
        atags = btags = ''
        cruncher.set_seqs(aelt, belt)
        for tag, ai1, ai2, bj1, bj2 in cruncher.get_opcodes():
            la, lb = (ai2 - ai1, bj2 - bj1)
            if tag == 'replace':
                atags += '^' * la
                btags += '^' * lb
            elif tag == 'delete':
                atags += '-' * la
            elif tag == 'insert':
                btags += '+' * lb
            elif tag == 'equal':
                atags += ' ' * la
                btags += ' ' * lb
            else:
                raise ValueError('unknown tag %r' % (tag,))
        yield from self._qformat(aelt, belt, atags, btags)
    else:
        yield ('  ' + aelt)
    yield from self._fancy_helper(a, best_i + 1, ahi, b, best_j + 1, bhi)