from lxml import etree
import sys
import re
import doctest
def _find_doctest_frame():
    import sys
    frame = sys._getframe(1)
    while frame:
        l = frame.f_locals
        if 'BOOM' in l:
            return frame
        frame = frame.f_back
    raise LookupError('Could not find doctest (only use this function *inside* a doctest)')