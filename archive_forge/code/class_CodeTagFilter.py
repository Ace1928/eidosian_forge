import re
from pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
from pygments.filter import Filter
from pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
from pygments.plugin import find_plugin_filters
class CodeTagFilter(Filter):
    """Highlight special code tags in comments and docstrings.

    Options accepted:

    `codetags` : list of strings
       A list of strings that are flagged as code tags.  The default is to
       highlight ``XXX``, ``TODO``, ``BUG`` and ``NOTE``.
    """

    def __init__(self, **options):
        Filter.__init__(self, **options)
        tags = get_list_opt(options, 'codetags', ['XXX', 'TODO', 'BUG', 'NOTE'])
        self.tag_re = re.compile('\\b(%s)\\b' % '|'.join([re.escape(tag) for tag in tags if tag]))

    def filter(self, lexer, stream):
        regex = self.tag_re
        for ttype, value in stream:
            if ttype in String.Doc or (ttype in Comment and ttype not in Comment.Preproc):
                for sttype, svalue in _replace_special(ttype, value, regex, Comment.Special):
                    yield (sttype, svalue)
            else:
                yield (ttype, value)