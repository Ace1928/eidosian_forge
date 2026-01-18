import os
import sys
import pytest
def _enable_faulthandler():
    """Enable faulthandler (if we can), so that we get tracebacks
    on segfaults.
    """
    try:
        import faulthandler
        faulthandler.enable()
        print('Faulthandler enabled')
    except Exception:
        print('Could not enable faulthandler')