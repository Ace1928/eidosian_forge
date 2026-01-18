def _vowelinstem(self):
    """Check if b[0: j + 1] contains a vowel letter.

        Returns
        -------
        bool

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.parsing.porter import PorterStemmer
            >>> p = PorterStemmer()
            >>> p.b = "gnsm"
            >>> p.j = 3
            >>> p._vowelinstem()
            False
            >>> p.b = "gensim"
            >>> p.j = 5
            >>> p._vowelinstem()
            True

        """
    return not all((self._cons(i) for i in range(self.j + 1)))