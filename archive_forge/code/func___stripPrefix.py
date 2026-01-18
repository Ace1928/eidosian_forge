import re
from nltk.stem.api import StemmerI
def __stripPrefix(self, word):
    """Remove prefix from a word.

        This function originally taken from Whoosh.

        """
    for prefix in ('kilo', 'micro', 'milli', 'intra', 'ultra', 'mega', 'nano', 'pico', 'pseudo'):
        if word.startswith(prefix):
            return word[len(prefix):]
    return word