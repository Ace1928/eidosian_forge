import re
from pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
from pygments.filter import Filter
from pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
from pygments.plugin import find_plugin_filters
class KeywordCaseFilter(Filter):
    """Convert keywords to lowercase or uppercase or capitalize them, which
    means first letter uppercase, rest lowercase.

    This can be useful e.g. if you highlight Pascal code and want to adapt the
    code to your styleguide.

    Options accepted:

    `case` : string
       The casing to convert keywords to. Must be one of ``'lower'``,
       ``'upper'`` or ``'capitalize'``.  The default is ``'lower'``.
    """

    def __init__(self, **options):
        Filter.__init__(self, **options)
        case = get_choice_opt(options, 'case', ['lower', 'upper', 'capitalize'], 'lower')
        self.convert = getattr(text_type, case)

    def filter(self, lexer, stream):
        for ttype, value in stream:
            if ttype in Keyword:
                yield (ttype, self.convert(value))
            else:
                yield (ttype, value)