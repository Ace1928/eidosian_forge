import os
import tempfile
import time
import re
def RaiseChemDraw():
    e = re.compile('^ChemDraw')
    RaiseWindowNamed(e)