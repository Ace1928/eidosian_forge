def DT_hit(self, count, ecrossing):
    """
        Count the crossing, using DT conventions.  Return True on the
        first hit if the count is odd and the crossing is shared by
        two components of the diagram.  As a side effect, set the
        flipped attribute on the first hit.
        """
    over = ecrossing.goes_over()
    if count % 2 == 0 and over:
        count = -count
    if self.hit1 == 0:
        self.hit1 = count
        sign = self.sign()
        if sign:
            self.flipped = over ^ (sign == 'RH')
        if count % 2 != 0 and self.comp1 != self.comp2:
            return True
    elif self.hit2 == 0:
        self.hit2 = count
    else:
        raise ValueError('Too many hits!')