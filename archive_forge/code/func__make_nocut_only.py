import re
def _make_nocut_only(self, nc, s1, ls=(), title=''):
    """Summarise non-cutting enzymes (PRIVATE).

        Return a formatted string of the non cutting enzymes.

        Arguments:
         - nc is a tuple or list of non cutting enzymes.
         - s1 is the sentence before the non cutting enzymes.
        """
    if not nc:
        return s1
    st = ''
    stringsite = s1 or '\n   Enzymes which do not cut the sequence.\n\n'
    Join = ''.join
    for key in sorted(nc):
        st = Join((st, str.ljust(str(key), self.NameWidth)))
        if len(st) > self.linesize:
            stringsite = Join((stringsite, st, '\n'))
            st = ''
    stringsite = Join((stringsite, st, '\n'))
    return stringsite