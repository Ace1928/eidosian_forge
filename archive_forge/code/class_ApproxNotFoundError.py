class ApproxNotFoundError(Error):

    def __init__(self, curve):
        message = 'no approximation found: %s' % curve
        super().__init__(message)
        self.curve = curve