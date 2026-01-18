import os
import warnings
class PyPWL:
    """Pure-python implementation of Personal Word List dictionary.
    This class emulates the PWL objects provided by PyEnchant, but
    implemented purely in python.
    """

    def __init__(self, pwl=None):
        """PyPWL constructor.
        This method takes as its only argument the name of a file
        containing the personal word list, one word per line.  Entries
        will be read from this file, and new entries will be written to
        it automatically.

        If <pwl> is not specified or None, the list is maintained in
        memory only.
        """
        self.provider = None
        self._words = Trie()
        if pwl is not None:
            self.pwl = os.path.abspath(pwl)
            self.tag = self.pwl
            pwl_f = open(pwl)
            for ln in pwl_f:
                word = ln.strip()
                self.add_to_session(word)
            pwl_f.close()
        else:
            self.pwl = None
            self.tag = 'PyPWL'

    def check(self, word):
        """Check spelling of a word.

        This method takes a word in the dictionary language and returns
        True if it is correctly spelled, and false otherwise.
        """
        res = self._words.search(word)
        return bool(res)

    def suggest(self, word):
        """Suggest possible spellings for a word.

        This method tries to guess the correct spelling for a given
        word, returning the possibilities in a list.
        """
        limit = 10
        maxdepth = 5
        depth = 0
        res = self._words.search(word, depth)
        while len(res) < limit and depth < maxdepth:
            depth += 1
            for w in self._words.search(word, depth):
                if w not in res:
                    res.append(w)
        return res[:limit]

    def add(self, word):
        """Add a word to the user's personal dictionary.
        For a PWL, this means appending it to the file.
        """
        if self.pwl is not None:
            pwl_f = open(self.pwl, 'a')
            pwl_f.write('%s\n' % (word.strip(),))
            pwl_f.close()
        self.add_to_session(word)

    def add_to_pwl(self, word):
        """Add a word to the user's personal dictionary.
        For a PWL, this means appending it to the file.
        """
        warnings.warn('PyPWL.add_to_pwl is deprecated, please use PyPWL.add', category=DeprecationWarning, stacklevel=2)
        self.add(word)

    def remove(self, word):
        """Add a word to the user's personal exclude list."""
        self._words.remove(word)
        if self.pwl is not None:
            pwl_f = open(self.pwl, 'wt')
            for w in self._words:
                pwl_f.write('%s\n' % (w.strip(),))
            pwl_f.close()

    def add_to_session(self, word):
        """Add a word to the session list."""
        self._words.insert(word)

    def store_replacement(self, mis, cor):
        """Store a replacement spelling for a miss-spelled word.

        This method makes a suggestion to the spellchecking engine that the
        miss-spelled word <mis> is in fact correctly spelled as <cor>.  Such
        a suggestion will typically mean that <cor> appears early in the
        list of suggested spellings offered for later instances of <mis>.
        """
        pass
    store_replacement._DOC_ERRORS = ['mis', 'mis']

    def is_added(self, word):
        """Check whether a word is in the personal word list."""
        return self.check(word)

    def is_removed(self, word):
        """Check whether a word is in the personal exclude list."""
        return False

    def _check_this(self, msg):
        pass

    def _free(self):
        pass