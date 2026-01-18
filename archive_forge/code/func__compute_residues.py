import cupy
def _compute_residues(poles, multiplicity, numerator):
    denominator_factors, _ = _compute_factors(poles, multiplicity)
    numerator = numerator.astype(poles.dtype)
    residues = []
    for pole, mult, factor in zip(poles, multiplicity, denominator_factors):
        if mult == 1:
            residues.append(cupy.polyval(numerator, pole) / cupy.polyval(factor, pole))
        else:
            numer = numerator.copy()
            monomial = cupy.r_[1, -pole]
            factor, d = _polydiv(factor, monomial)
            block = []
            for _ in range(int(mult)):
                numer, n = _polydiv(numer, monomial)
                r = n[0] / d[0]
                numer = cupy.polysub(numer, r * factor)
                block.append(r)
            residues.extend(reversed(block))
    return cupy.asarray(residues)