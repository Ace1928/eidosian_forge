import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
class MentionFilter(Filter):
    """Filter skipping over @mention.
    This filter skips any words matching the following regular expression:

           (\\A|\\s)@(\\w+)

    That is, any words that are @mention.
    """
    _DOC_ERRORS = ['zA']
    _pattern = re.compile('(\\A|\\s)@(\\w+)')

    def _skip(self, word):
        if self._pattern.match(word):
            return True
        return False