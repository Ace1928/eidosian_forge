def _doublec(self, j):
    """Check if b[j - 1: j + 1] contain a double consonant letter.

        Parameters
        ----------
        j : int
            Index for `b`

        Returns
        -------
        bool

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.parsing.porter import PorterStemmer
            >>> p = PorterStemmer()
            >>> p.b = "real"
            >>> p.j = 3
            >>> p._doublec(3)
            False
            >>> p.b = "really"
            >>> p.j = 5
            >>> p._doublec(4)
            True

        """
    return j > 0 and self.b[j] == self.b[j - 1] and self._cons(j)