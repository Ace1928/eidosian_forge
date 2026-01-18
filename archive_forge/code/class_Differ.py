from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
class Differ:
    """
    Differ is a class for comparing sequences of lines of text, and
    producing human-readable differences or deltas.  Differ uses
    SequenceMatcher both to compare sequences of lines, and to compare
    sequences of characters within similar (near-matching) lines.

    Each line of a Differ delta begins with a two-letter code:

        '- '    line unique to sequence 1
        '+ '    line unique to sequence 2
        '  '    line common to both sequences
        '? '    line not present in either input sequence

    Lines beginning with '? ' attempt to guide the eye to intraline
    differences, and were not present in either input sequence.  These lines
    can be confusing if the sequences contain tab characters.

    Note that Differ makes no claim to produce a *minimal* diff.  To the
    contrary, minimal diffs are often counter-intuitive, because they synch
    up anywhere possible, sometimes accidental matches 100 pages apart.
    Restricting synch points to contiguous matches preserves some notion of
    locality, at the occasional cost of producing a longer diff.

    Example: Comparing two texts.

    First we set up the texts, sequences of individual single-line strings
    ending with newlines (such sequences can also be obtained from the
    `readlines()` method of file-like objects):

    >>> text1 = '''  1. Beautiful is better than ugly.
    ...   2. Explicit is better than implicit.
    ...   3. Simple is better than complex.
    ...   4. Complex is better than complicated.
    ... '''.splitlines(keepends=True)
    >>> len(text1)
    4
    >>> text1[0][-1]
    '\\n'
    >>> text2 = '''  1. Beautiful is better than ugly.
    ...   3.   Simple is better than complex.
    ...   4. Complicated is better than complex.
    ...   5. Flat is better than nested.
    ... '''.splitlines(keepends=True)

    Next we instantiate a Differ object:

    >>> d = Differ()

    Note that when instantiating a Differ object we may pass functions to
    filter out line and character 'junk'.  See Differ.__init__ for details.

    Finally, we compare the two:

    >>> result = list(d.compare(text1, text2))

    'result' is a list of strings, so let's pretty-print it:

    >>> from pprint import pprint as _pprint
    >>> _pprint(result)
    ['    1. Beautiful is better than ugly.\\n',
     '-   2. Explicit is better than implicit.\\n',
     '-   3. Simple is better than complex.\\n',
     '+   3.   Simple is better than complex.\\n',
     '?     ++\\n',
     '-   4. Complex is better than complicated.\\n',
     '?            ^                     ---- ^\\n',
     '+   4. Complicated is better than complex.\\n',
     '?           ++++ ^                      ^\\n',
     '+   5. Flat is better than nested.\\n']

    As a single multi-line string it looks like this:

    >>> print(''.join(result), end="")
        1. Beautiful is better than ugly.
    -   2. Explicit is better than implicit.
    -   3. Simple is better than complex.
    +   3.   Simple is better than complex.
    ?     ++
    -   4. Complex is better than complicated.
    ?            ^                     ---- ^
    +   4. Complicated is better than complex.
    ?           ++++ ^                      ^
    +   5. Flat is better than nested.
    """

    def __init__(self, linejunk=None, charjunk=None):
        """
        Construct a text differencer, with optional filters.

        The two optional keyword parameters are for filter functions:

        - `linejunk`: A function that should accept a single string argument,
          and return true iff the string is junk. The module-level function
          `IS_LINE_JUNK` may be used to filter out lines without visible
          characters, except for at most one splat ('#').  It is recommended
          to leave linejunk None; the underlying SequenceMatcher class has
          an adaptive notion of "noise" lines that's better than any static
          definition the author has ever been able to craft.

        - `charjunk`: A function that should accept a string of length 1. The
          module-level function `IS_CHARACTER_JUNK` may be used to filter out
          whitespace characters (a blank or tab; **note**: bad idea to include
          newline in this!).  Use of IS_CHARACTER_JUNK is recommended.
        """
        self.linejunk = linejunk
        self.charjunk = charjunk

    def compare(self, a, b):
        """
        Compare two sequences of lines; generate the resulting delta.

        Each sequence must contain individual single-line strings ending with
        newlines. Such sequences can be obtained from the `readlines()` method
        of file-like objects.  The delta generated also consists of newline-
        terminated strings, ready to be printed as-is via the writelines()
        method of a file-like object.

        Example:

        >>> print(''.join(Differ().compare('one\\ntwo\\nthree\\n'.splitlines(True),
        ...                                'ore\\ntree\\nemu\\n'.splitlines(True))),
        ...       end="")
        - one
        ?  ^
        + ore
        ?  ^
        - two
        - three
        ?  -
        + tree
        + emu
        """
        cruncher = SequenceMatcher(self.linejunk, a, b)
        for tag, alo, ahi, blo, bhi in cruncher.get_opcodes():
            if tag == 'replace':
                g = self._fancy_replace(a, alo, ahi, b, blo, bhi)
            elif tag == 'delete':
                g = self._dump('-', a, alo, ahi)
            elif tag == 'insert':
                g = self._dump('+', b, blo, bhi)
            elif tag == 'equal':
                g = self._dump(' ', a, alo, ahi)
            else:
                raise ValueError('unknown tag %r' % (tag,))
            yield from g

    def _dump(self, tag, x, lo, hi):
        """Generate comparison results for a same-tagged range."""
        for i in range(lo, hi):
            yield ('%s %s' % (tag, x[i]))

    def _plain_replace(self, a, alo, ahi, b, blo, bhi):
        assert alo < ahi and blo < bhi
        if bhi - blo < ahi - alo:
            first = self._dump('+', b, blo, bhi)
            second = self._dump('-', a, alo, ahi)
        else:
            first = self._dump('-', a, alo, ahi)
            second = self._dump('+', b, blo, bhi)
        for g in (first, second):
            yield from g

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

    def _fancy_helper(self, a, alo, ahi, b, blo, bhi):
        g = []
        if alo < ahi:
            if blo < bhi:
                g = self._fancy_replace(a, alo, ahi, b, blo, bhi)
            else:
                g = self._dump('-', a, alo, ahi)
        elif blo < bhi:
            g = self._dump('+', b, blo, bhi)
        yield from g

    def _qformat(self, aline, bline, atags, btags):
        """
        Format "?" output and deal with tabs.

        Example:

        >>> d = Differ()
        >>> results = d._qformat('\\tabcDefghiJkl\\n', '\\tabcdefGhijkl\\n',
        ...                      '  ^ ^  ^      ', '  ^ ^  ^      ')
        >>> for line in results: print(repr(line))
        ...
        '- \\tabcDefghiJkl\\n'
        '? \\t ^ ^  ^\\n'
        '+ \\tabcdefGhijkl\\n'
        '? \\t ^ ^  ^\\n'
        """
        atags = _keep_original_ws(aline, atags).rstrip()
        btags = _keep_original_ws(bline, btags).rstrip()
        yield ('- ' + aline)
        if atags:
            yield f'? {atags}\n'
        yield ('+ ' + bline)
        if btags:
            yield f'? {btags}\n'