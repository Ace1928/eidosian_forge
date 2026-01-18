def _step3(self):
    """Deal with -ic-, -full, -ness etc. Similar strategy to _step2."""
    ch = self.b[self.k]
    if ch == 'e':
        if self._ends('icate'):
            self._r('ic')
        elif self._ends('ative'):
            self._r('')
        elif self._ends('alize'):
            self._r('al')
    elif ch == 'i':
        if self._ends('iciti'):
            self._r('ic')
    elif ch == 'l':
        if self._ends('ical'):
            self._r('ic')
        elif self._ends('ful'):
            self._r('')
    elif ch == 's':
        if self._ends('ness'):
            self._r('')