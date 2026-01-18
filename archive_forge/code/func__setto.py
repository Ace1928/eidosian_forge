def _setto(self, s):
    """Append `s` to `b`, adjusting `k`.

        Parameters
        ----------
        s : str

        """
    self.b = self.b[:self.j + 1] + s
    self.k = len(self.b) - 1