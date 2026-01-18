class SingularityError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = str(self.holonomic)
        s += ' has a singularity at %s.' % self.x0
        return s