import os
import tempfile
import time
import re
def CDXConvert(inData, inFormat, outFormat):
    """converts the data passed in from one format to another

    inFormat should be one of the following:
       chemical/x-cdx                   chemical/cdx 
       chemical/x-daylight-smiles       chemical/daylight-smiles 
       chemical/x-mdl-isis              chemical/mdl-isis 
       chemical/x-mdl-molfile           chemical/mdl-molfile 
       chemical/x-mdl-rxn               chemical/mdl-rxn 
       chemical/x-mdl-tgf               chemical/mdl-tgf 
       chemical/x-questel-F1  
       chemical/x-questel-F1-query 

    outFormat should be one of the preceding or:
       image/x-png                      image/png 
       image/x-wmf                      image/wmf 
       image/tiff  
       application/postscript  
       image/gif  
  """
    global theObjs, theDoc
    if cdApp is None:
        StartChemDraw()
    if theObjs is None:
        if theDoc is None:
            theDoc = cdApp.Documents.Add()
        theObjs = theDoc.Objects
    theObjs.SetData(inFormat, inData, pythoncom.Missing)
    outD = theObjs.GetData(outFormat)
    theObjs.Clear()
    return outD