from pecan.compat import getargspec as _getargspec
def _cfg(f):
    if not hasattr(f, '_pecan'):
        f._pecan = {}
    return f._pecan