import cupy
def _compute_factors(roots, multiplicity, include_powers=False):
    """Compute the total polynomial divided by factors for each root."""
    current = cupy.array([1])
    suffixes = [current]
    for pole, mult in zip(roots[-1:0:-1], multiplicity[-1:0:-1]):
        monomial = cupy.r_[1, -pole]
        for _ in range(int(mult)):
            current = cupy.polymul(current, monomial)
        suffixes.append(current)
    suffixes = suffixes[::-1]
    factors = []
    current = cupy.array([1])
    for pole, mult, suffix in zip(roots, multiplicity, suffixes):
        monomial = cupy.r_[1, -pole]
        block = []
        for i in range(int(mult)):
            if i == 0 or include_powers:
                block.append(cupy.polymul(current, suffix))
            current = cupy.polymul(current, monomial)
        factors.extend(reversed(block))
    return (factors, current)