def _step1c(self):
    """Turn terminal 'y' to 'i' when there is another vowel in the stem."""
    if self._ends('y') and self._vowelinstem():
        self.b = self.b[:self.k] + 'i'