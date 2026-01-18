import os
import tempfile
import time
import re
def CDXClean(inData, inFormat, outFormat):
    """calls the CDXLib Clean function on the data passed in.

    CDXLib_Clean attempts to clean (prettify) the data before
    doing an output conversion.  It can be thought of as CDXConvert++.

    CDXClean supports the same input and output specifiers as CDXConvert
    (see above)

  """
    global cdApp, theDoc, theObjs, selectItem, cleanItem
    if cdApp is None:
        StartChemDraw()
    if theObjs is None:
        if theDoc is None:
            theDoc = cdApp.Documents.Add()
        theObjs = theDoc.Objects
    theObjs.SetData(inFormat, inData, pythoncom.Missing)
    theObjs.Select()
    cleanItem.Execute()
    outD = theObjs.GetData(outFormat)
    theObjs.Clear()
    return outD