import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
class WikiWordFilter(Filter):
    """Filter skipping over WikiWords.
    This filter skips any words matching the following regular expression:

           ^([A-Z]\\w+[A-Z]+\\w+)

    That is, any words that are WikiWords.
    """
    _pattern = re.compile('^([A-Z]\\w+[A-Z]+\\w+)')

    def _skip(self, word):
        if self._pattern.match(word):
            return True
        return False