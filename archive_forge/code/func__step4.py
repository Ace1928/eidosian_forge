def _step4(self):
    """Takes off -ant, -ence etc., in context <c>vcvc<v>."""
    ch = self.b[self.k - 1]
    if ch == 'a':
        if not self._ends('al'):
            return
    elif ch == 'c':
        if not self._ends('ance') and (not self._ends('ence')):
            return
    elif ch == 'e':
        if not self._ends('er'):
            return
    elif ch == 'i':
        if not self._ends('ic'):
            return
    elif ch == 'l':
        if not self._ends('able') and (not self._ends('ible')):
            return
    elif ch == 'n':
        if self._ends('ant'):
            pass
        elif self._ends('ement'):
            pass
        elif self._ends('ment'):
            pass
        elif self._ends('ent'):
            pass
        else:
            return
    elif ch == 'o':
        if self._ends('ion') and self.b[self.j] in 'st':
            pass
        elif self._ends('ou'):
            pass
        else:
            return
    elif ch == 's':
        if not self._ends('ism'):
            return
    elif ch == 't':
        if not self._ends('ate') and (not self._ends('iti')):
            return
    elif ch == 'u':
        if not self._ends('ous'):
            return
    elif ch == 'v':
        if not self._ends('ive'):
            return
    elif ch == 'z':
        if not self._ends('ize'):
            return
    else:
        return
    if self._m() > 1:
        self.k = self.j