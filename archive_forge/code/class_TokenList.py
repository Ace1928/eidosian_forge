import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
class TokenList(list):
    token_type = None
    syntactic_break = True
    ew_combine_allowed = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.defects = []

    def __str__(self):
        return ''.join((str(x) for x in self))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, super().__repr__())

    @property
    def value(self):
        return ''.join((x.value for x in self if x.value))

    @property
    def all_defects(self):
        return sum((x.all_defects for x in self), self.defects)

    def startswith_fws(self):
        return self[0].startswith_fws()

    @property
    def as_ew_allowed(self):
        """True if all top level tokens of this part may be RFC2047 encoded."""
        return all((part.as_ew_allowed for part in self))

    @property
    def comments(self):
        comments = []
        for token in self:
            comments.extend(token.comments)
        return comments

    def fold(self, *, policy):
        return _refold_parse_tree(self, policy=policy)

    def pprint(self, indent=''):
        print(self.ppstr(indent=indent))

    def ppstr(self, indent=''):
        return '\n'.join(self._pp(indent=indent))

    def _pp(self, indent=''):
        yield '{}{}/{}('.format(indent, self.__class__.__name__, self.token_type)
        for token in self:
            if not hasattr(token, '_pp'):
                yield (indent + '    !! invalid element in token list: {!r}'.format(token))
            else:
                yield from token._pp(indent + '    ')
        if self.defects:
            extra = ' Defects: {}'.format(self.defects)
        else:
            extra = ''
        yield '{}){}'.format(indent, extra)