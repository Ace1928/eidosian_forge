def _cons(self, i):
    """Check if b[i] is a consonant letter.

        Parameters
        ----------
        i : int
            Index for `b`.

        Returns
        -------
        bool

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.parsing.porter import PorterStemmer
            >>> p = PorterStemmer()
            >>> p.b = "hi"
            >>> p._cons(1)
            False
            >>> p.b = "meow"
            >>> p._cons(3)
            True

        """
    ch = self.b[i]
    if ch in 'aeiou':
        return False
    if ch == 'y':
        return i == 0 or not self._cons(i - 1)
    return True