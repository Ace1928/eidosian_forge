class NotHolonomicError(BaseHolonomicError):

    def __init__(self, m):
        self.m = m

    def __str__(self):
        return self.m