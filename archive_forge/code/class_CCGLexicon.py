import re
from collections import defaultdict
from nltk.ccg.api import CCGVar, Direction, FunctionalCategory, PrimitiveCategory
from nltk.internals import deprecated
from nltk.sem.logic import Expression
class CCGLexicon:
    """
    Class representing a lexicon for CCG grammars.

    * `primitives`: The list of primitive categories for the lexicon
    * `families`: Families of categories
    * `entries`: A mapping of words to possible categories
    """

    def __init__(self, start, primitives, families, entries):
        self._start = PrimitiveCategory(start)
        self._primitives = primitives
        self._families = families
        self._entries = entries

    def categories(self, word):
        """
        Returns all the possible categories for a word
        """
        return self._entries[word]

    def start(self):
        """
        Return the target category for the parser
        """
        return self._start

    def __str__(self):
        """
        String representation of the lexicon. Used for debugging.
        """
        string = ''
        first = True
        for ident in sorted(self._entries):
            if not first:
                string = string + '\n'
            string = string + ident + ' => '
            first = True
            for cat in self._entries[ident]:
                if not first:
                    string = string + ' | '
                else:
                    first = False
                string = string + '%s' % cat
        return string