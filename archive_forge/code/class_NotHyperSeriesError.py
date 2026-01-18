class NotHyperSeriesError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = 'Power series expansion of '
        s += str(self.holonomic)
        s += ' about %s is not hypergeometric' % self.x0
        return s