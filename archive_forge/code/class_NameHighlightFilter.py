import re
from pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
from pygments.filter import Filter
from pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
from pygments.plugin import find_plugin_filters
class NameHighlightFilter(Filter):
    """Highlight a normal Name (and Name.*) token with a different token type.

    Example::

        filter = NameHighlightFilter(
            names=['foo', 'bar', 'baz'],
            tokentype=Name.Function,
        )

    This would highlight the names "foo", "bar" and "baz"
    as functions. `Name.Function` is the default token type.

    Options accepted:

    `names` : list of strings
      A list of names that should be given the different token type.
      There is no default.
    `tokentype` : TokenType or string
      A token type or a string containing a token type name that is
      used for highlighting the strings in `names`.  The default is
      `Name.Function`.
    """

    def __init__(self, **options):
        Filter.__init__(self, **options)
        self.names = set(get_list_opt(options, 'names', []))
        tokentype = options.get('tokentype')
        if tokentype:
            self.tokentype = string_to_tokentype(tokentype)
        else:
            self.tokentype = Name.Function

    def filter(self, lexer, stream):
        for ttype, value in stream:
            if ttype in Name and value in self.names:
                yield (self.tokentype, value)
            else:
                yield (ttype, value)