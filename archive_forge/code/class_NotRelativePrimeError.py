from rsa._compat import zip
class NotRelativePrimeError(ValueError):

    def __init__(self, a, b, d, msg=None):
        super(NotRelativePrimeError, self).__init__(msg or '%d and %d are not relatively prime, divider=%i' % (a, b, d))
        self.a = a
        self.b = b
        self.d = d