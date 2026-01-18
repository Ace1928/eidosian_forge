import re
from pyomo.common.deprecation import deprecated
from pyomo.core.base.componentuid import ComponentUID
class _CharMapper(object):

    def __init__(self, preserve, translate, other):
        """
        Arguments::
           preserve: a string of characters to preserve
           translate: a dict or key/value list of characters to translate
           other: the character to return for all characters not in
                  preserve or translate
        """
        self.table = {k if isinstance(k, int) else ord(k): v for k, v in dict(translate).items()}
        for c in preserve:
            _c = ord(c)
            if _c in self.table and self.table[_c] != c:
                raise RuntimeError("Duplicate character '%s' appears in both translate table and preserve list" % (c,))
            self.table[_c] = c
        self.other = other

    def __getitem__(self, c):
        try:
            return self.table[c]
        except:
            self.table[c] = self.other
            return self.other

    def make_table(self):
        return ''.join((self[i] for i in range(256)))