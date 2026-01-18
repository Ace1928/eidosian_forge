def _step1ab(self):
    """Get rid of plurals and -ed or -ing.

           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat

           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable

           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess

           meetings  ->  meet

        """
    if self.b[self.k] == 's':
        if self._ends('sses'):
            self.k -= 2
        elif self._ends('ies'):
            self._setto('i')
        elif self.b[self.k - 1] != 's':
            self.k -= 1
    if self._ends('eed'):
        if self._m() > 0:
            self.k -= 1
    elif (self._ends('ed') or self._ends('ing')) and self._vowelinstem():
        self.k = self.j
        if self._ends('at'):
            self._setto('ate')
        elif self._ends('bl'):
            self._setto('ble')
        elif self._ends('iz'):
            self._setto('ize')
        elif self._doublec(self.k):
            if self.b[self.k - 1] not in 'lsz':
                self.k -= 1
        elif self._m() == 1 and self._cvc(self.k):
            self._setto('e')