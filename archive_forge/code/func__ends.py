def _ends(self, s):
    """Check if b[: k + 1] ends with `s`.

        Parameters
        ----------
        s : str

        Returns
        -------
        bool

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.parsing.porter import PorterStemmer
            >>> p = PorterStemmer()
            >>> p.b = "cowboy"
            >>> p.j = 5
            >>> p.k = 2
            >>> p._ends("cow")
            True

        """
    if s[-1] != self.b[self.k]:
        return False
    length = len(s)
    if length > self.k + 1:
        return False
    if self.b[self.k - length + 1:self.k + 1] != s:
        return False
    self.j = self.k - length
    return True