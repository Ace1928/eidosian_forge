import re
from nltk.stem.api import StemmerI
def _is_consonant(self, word, i):
    """Returns True if word[i] is a consonant, False otherwise

        A consonant is defined in the paper as follows:

            A consonant in a word is a letter other than A, E, I, O or
            U, and other than Y preceded by a consonant. (The fact that
            the term `consonant' is defined to some extent in terms of
            itself does not make it ambiguous.) So in TOY the consonants
            are T and Y, and in SYZYGY they are S, Z and G. If a letter
            is not a consonant it is a vowel.
        """
    if word[i] in self.vowels:
        return False
    if word[i] == 'y':
        if i == 0:
            return True
        else:
            return not self._is_consonant(word, i - 1)
    return True