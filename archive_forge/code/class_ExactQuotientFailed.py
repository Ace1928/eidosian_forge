from sympy.utilities import public
@public
class ExactQuotientFailed(BasePolynomialError):

    def __init__(self, f, g, dom=None):
        self.f, self.g, self.dom = (f, g, dom)

    def __str__(self):
        from sympy.printing.str import sstr
        if self.dom is None:
            return '%s does not divide %s' % (sstr(self.g), sstr(self.f))
        else:
            return '%s does not divide %s in %s' % (sstr(self.g), sstr(self.f), sstr(self.dom))

    def new(self, f, g):
        return self.__class__(f, g, self.dom)