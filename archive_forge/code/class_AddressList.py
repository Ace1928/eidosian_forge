import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
class AddressList(TokenList):
    token_type = 'address-list'

    @property
    def addresses(self):
        return [x for x in self if x.token_type == 'address']

    @property
    def mailboxes(self):
        return sum((x.mailboxes for x in self if x.token_type == 'address'), [])

    @property
    def all_mailboxes(self):
        return sum((x.all_mailboxes for x in self if x.token_type == 'address'), [])