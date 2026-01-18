import re
def _patch_quantities(pq):
    if not hasattr(pq.dimensionality.Dimensionality, 'html'):
        pq.dimensionality.Dimensionality.html = property(lambda self: format_units_html(self))
    a = pq.UncertainQuantity([1, 2], pq.m, [0.1, 0.2])
    if (-a).uncertainty[0] != a.uncertainty[0]:
        pq.UncertainQuantity.__neg__ = lambda self: self * -1
    a = pq.UncertainQuantity([1, 2], pq.m, [0.1, 0.2])
    assert (-a).uncertainty[0] == (a * -1).uncertainty[0]
    _patch_pow0_py35(pq)
    assert (3 * pq.m) ** 0 == 1 * pq.dimensionless