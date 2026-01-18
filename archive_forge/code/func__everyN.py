from fontTools.cffLib import maxStackLimit
def _everyN(el, n):
    """Group the list el into groups of size n"""
    if len(el) % n != 0:
        raise ValueError(el)
    for i in range(0, len(el), n):
        yield el[i:i + n]