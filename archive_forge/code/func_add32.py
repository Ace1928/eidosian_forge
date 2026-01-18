import reportlab
def add32(x, y):
    """Calculate (x + y) modulo 2**32"""
    return x + y & 4294967295