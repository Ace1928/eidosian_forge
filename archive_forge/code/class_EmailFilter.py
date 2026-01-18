import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
class EmailFilter(Filter):
    """Filter skipping over email addresses.
    This filter skips any words matching the following regular expression:

           ^.+@[^\\.].*\\.[a-z]{2,}$

    That is, any words that resemble email addresses.
    """
    _pattern = re.compile('^.+@[^\\.].*\\.[a-z]{2,}$')

    def _skip(self, word):
        if self._pattern.match(word):
            return True
        return False