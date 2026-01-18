import os
import tempfile
import time
import re
def SaveChemDrawDoc(fileName='save.cdx'):
    """force chemdraw to save the active document

  NOTE: the extension of the filename will determine the format
   used to save the file.
  """
    d = cdApp.ActiveDocument
    d.SaveAs(fileName)