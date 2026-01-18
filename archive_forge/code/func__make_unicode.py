import sys
def _make_unicode(b):
    if hasattr(b, 'decode'):
        return b.decode()
    else:
        return str(b)