import math
def _partial_sums(vals):
    """Adds up the items in the input, yielding partial sums along the way."""
    total = 0
    for v in vals:
        yield total
        total += v
    yield total