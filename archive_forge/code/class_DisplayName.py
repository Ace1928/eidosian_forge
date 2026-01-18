import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
class DisplayName(Phrase):
    token_type = 'display-name'
    ew_combine_allowed = False

    @property
    def display_name(self):
        res = TokenList(self)
        if len(res) == 0:
            return res.value
        if res[0].token_type == 'cfws':
            res.pop(0)
        elif res[0][0].token_type == 'cfws':
            res[0] = TokenList(res[0][1:])
        if res[-1].token_type == 'cfws':
            res.pop()
        elif res[-1][-1].token_type == 'cfws':
            res[-1] = TokenList(res[-1][:-1])
        return res.value

    @property
    def value(self):
        quote = False
        if self.defects:
            quote = True
        else:
            for x in self:
                if x.token_type == 'quoted-string':
                    quote = True
        if len(self) != 0 and quote:
            pre = post = ''
            if self[0].token_type == 'cfws' or self[0][0].token_type == 'cfws':
                pre = ' '
            if self[-1].token_type == 'cfws' or self[-1][-1].token_type == 'cfws':
                post = ' '
            return pre + quote_string(self.display_name) + post
        else:
            return super().value