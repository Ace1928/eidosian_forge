def _step2(self):
    """Map double suffices to single ones.

        So, -ization ( = -ize plus -ation) maps to -ize etc. Note that the
        string before the suffix must give _m() > 0.

        """
    ch = self.b[self.k - 1]
    if ch == 'a':
        if self._ends('ational'):
            self._r('ate')
        elif self._ends('tional'):
            self._r('tion')
    elif ch == 'c':
        if self._ends('enci'):
            self._r('ence')
        elif self._ends('anci'):
            self._r('ance')
    elif ch == 'e':
        if self._ends('izer'):
            self._r('ize')
    elif ch == 'l':
        if self._ends('bli'):
            self._r('ble')
        elif self._ends('alli'):
            self._r('al')
        elif self._ends('entli'):
            self._r('ent')
        elif self._ends('eli'):
            self._r('e')
        elif self._ends('ousli'):
            self._r('ous')
    elif ch == 'o':
        if self._ends('ization'):
            self._r('ize')
        elif self._ends('ation'):
            self._r('ate')
        elif self._ends('ator'):
            self._r('ate')
    elif ch == 's':
        if self._ends('alism'):
            self._r('al')
        elif self._ends('iveness'):
            self._r('ive')
        elif self._ends('fulness'):
            self._r('ful')
        elif self._ends('ousness'):
            self._r('ous')
    elif ch == 't':
        if self._ends('aliti'):
            self._r('al')
        elif self._ends('iviti'):
            self._r('ive')
        elif self._ends('biliti'):
            self._r('ble')
    elif ch == 'g':
        if self._ends('logi'):
            self._r('log')