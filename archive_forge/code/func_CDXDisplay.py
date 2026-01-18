import os
import tempfile
import time
import re
def CDXDisplay(inData, inFormat='chemical/cdx', clear=1):
    """ displays the data in Chemdraw """
    global cdApp, theDoc, theObjs, selectItem, cleanItem, centerItem
    if cdApp is None:
        StartChemDraw()
    try:
        theDoc.Activate()
    except Exception:
        ReactivateChemDraw()
        theObjs = theDoc.Objects
    if clear:
        theObjs.Clear()
    theObjs.SetData(inFormat, inData, pythoncom.Missing)
    return