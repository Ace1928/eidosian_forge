import sys
def _tryimport(*names):
    """ return the first successfully imported module. """
    assert names
    for name in names:
        try:
            __import__(name)
        except ImportError:
            excinfo = sys.exc_info()
        else:
            return sys.modules[name]
    _reraise(*excinfo)