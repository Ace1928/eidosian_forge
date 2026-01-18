from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def _find_field_uncached(self, prec, degree, verbosity=False):

    def min_poly(z):
        return z.min_polynomial(prec, degree)

    def min_poly_deg(z):
        return z.min_polynomial().degree()

    def message(*args):
        if verbosity:
            print(*args)
    elts = self.list()
    z = ApproximateAlgebraicNumber(1)
    z.min_polynomial()
    exact_elts = []
    for i, elt in enumerate(elts):
        exact_elt = z.express(elt, prec)
        if exact_elt is not None:
            exact_elts.append(exact_elt)
        else:
            if min_poly(elt) is None:
                message('Bailing: no minimal polynomial found for newly considered element', elt)
                return None
            found = False
            candidates = [elt, z + elt, z - elt, z * elt, elt + elt + z, z + z + elt, elt + elt - z, z + z - elt, z + z * elt, elt + elt * z]
            for w in candidates:
                if min_poly(w) is None:
                    message('Skipping: no minimal polynomial found for possible primitive element', elt)
                elif min_poly_deg(w) >= min_poly_deg(z) and w.can_express(z, prec) and w.can_express(elt, prec):
                    exact_elts = w.express_several(elts[:i + 1], prec)
                    if exact_elts is None:
                        message("Bailing: Couldn't express everythingin terms of primitive element")
                        return None
                    found, z = (True, w)
                    break
            if not found:
                message("Bailing: Couldn't find primitive element for larger field")
                return None
    field = z.number_field()
    exact_elts = [field(exact_elt) for exact_elt in exact_elts]
    return (field, z, exact_elts)