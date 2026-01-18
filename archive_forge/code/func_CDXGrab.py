import os
import tempfile
import time
import re
def CDXGrab(outFormat='chemical/x-mdl-molfile'):
    """ returns the contents of the active chemdraw document

  """
    global cdApp, theDoc
    if cdApp is None:
        res = ''
    else:
        cdApp.Visible = 1
        if not cdApp.ActiveDocument:
            ReactivateChemDraw()
        try:
            res = cdApp.ActiveDocument.Objects.GetData(outFormat)
        except Exception:
            res = ''
    return res