import math
def _differences(weights):
    """Iterates over the input yielding differences between adjacent items."""
    previous_weight = None
    have_previous_weight = False
    for w in weights:
        if have_previous_weight:
            yield (w - previous_weight)
        previous_weight = w
        have_previous_weight = True