import re
from pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
from pygments.filter import Filter
from pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
from pygments.plugin import find_plugin_filters
class GobbleFilter(Filter):
    """Gobbles source code lines (eats initial characters).

    This filter drops the first ``n`` characters off every line of code.  This
    may be useful when the source code fed to the lexer is indented by a fixed
    amount of space that isn't desired in the output.

    Options accepted:

    `n` : int
       The number of characters to gobble.

    .. versionadded:: 1.2
    """

    def __init__(self, **options):
        Filter.__init__(self, **options)
        self.n = get_int_opt(options, 'n', 0)

    def gobble(self, value, left):
        if left < len(value):
            return (value[left:], 0)
        else:
            return (u'', left - len(value))

    def filter(self, lexer, stream):
        n = self.n
        left = n
        for ttype, value in stream:
            parts = value.split('\n')
            parts[0], left = self.gobble(parts[0], left)
            for i in range(1, len(parts)):
                parts[i], left = self.gobble(parts[i], n)
            value = u'\n'.join(parts)
            if value != '':
                yield (ttype, value)