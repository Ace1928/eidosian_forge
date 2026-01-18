from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def _pprint_tree(self, max_depth=None, depth=0, f=None, _pre=''):
    """Pretty-print the object tree."""
    token_count = len(self.tokens)
    for idx, token in enumerate(self.tokens):
        cls = token._get_repr_name()
        value = token._get_repr_value()
        last = idx == token_count - 1
        pre = u'`- ' if last else u'|- '
        q = u'"' if value.startswith("'") and value.endswith("'") else u"'"
        print(u'{_pre}{pre}{idx} {cls} {q}{value}{q}'.format(**locals()), file=f)
        if token.is_group and (max_depth is None or depth < max_depth):
            parent_pre = u'   ' if last else u'|  '
            token._pprint_tree(max_depth, depth + 1, f, _pre + parent_pre)