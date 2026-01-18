import re
from nltk.stem.api import StemmerI
def _step5a(self, word):
    """Implements Step 5a from "An algorithm for suffix stripping"

        From the paper:

        Step 5a

            (m>1) E     ->                  probate        ->  probat
                                            rate           ->  rate
            (m=1 and not *o) E ->           cease          ->  ceas
        """
    if word.endswith('e'):
        stem = self._replace_suffix(word, 'e', '')
        if self._measure(stem) > 1:
            return stem
        if self._measure(stem) == 1 and (not self._ends_cvc(stem)):
            return stem
    return word