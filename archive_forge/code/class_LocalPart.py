import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
class LocalPart(TokenList):
    token_type = 'local-part'
    as_ew_allowed = False

    @property
    def value(self):
        if self[0].token_type == 'quoted-string':
            return self[0].quoted_value
        else:
            return self[0].value

    @property
    def local_part(self):
        res = [DOT]
        last = DOT
        last_is_tl = False
        for tok in self[0] + [DOT]:
            if tok.token_type == 'cfws':
                continue
            if last_is_tl and tok.token_type == 'dot' and (last[-1].token_type == 'cfws'):
                res[-1] = TokenList(last[:-1])
            is_tl = isinstance(tok, TokenList)
            if is_tl and last.token_type == 'dot' and (tok[0].token_type == 'cfws'):
                res.append(TokenList(tok[1:]))
            else:
                res.append(tok)
            last = res[-1]
            last_is_tl = is_tl
        res = TokenList(res[1:-1])
        return res.value