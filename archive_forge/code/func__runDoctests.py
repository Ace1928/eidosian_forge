import math
def _runDoctests(verbose=None):
    import doctest
    import sys
    failed, _ = doctest.testmod(optionflags=doctest.ELLIPSIS, verbose=verbose)
    sys.exit(failed)