import re
from nltk.stem.api import StemmerI
def _ends_cvc(self, word):
    """Implements condition *o from the paper

        From the paper:

            *o  - the stem ends cvc, where the second c is not W, X or Y
                  (e.g. -WIL, -HOP).
        """
    return len(word) >= 3 and self._is_consonant(word, len(word) - 3) and (not self._is_consonant(word, len(word) - 2)) and self._is_consonant(word, len(word) - 1) and (word[-1] not in ('w', 'x', 'y')) or (self.mode == self.NLTK_EXTENSIONS and len(word) == 2 and (not self._is_consonant(word, 0)) and self._is_consonant(word, 1))