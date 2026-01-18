import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
class URLFilter(Filter):
    """Filter skipping over URLs.
    This filter skips any words matching the following regular expression:

           ^[a-zA-Z]+:\\/\\/[^\\s].*

    That is, any words that are URLs.
    """
    _DOC_ERRORS = ['zA']
    _pattern = re.compile('^[a-zA-Z]+:\\/\\/[^\\s].*')

    def _skip(self, word):
        if self._pattern.match(word):
            return True
        return False