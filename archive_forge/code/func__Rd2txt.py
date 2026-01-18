import os
from collections import namedtuple
import re
import sqlite3
import typing
import warnings
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects.packages_utils import (get_packagepath,
from collections import OrderedDict
def _Rd2txt(section_doc):
    tempfilename = rinterface.baseenv['tempfile']()[0]
    filecon = rinterface.baseenv['file'](tempfilename, open='w')
    try:
        tools_ns['Rd2txt'](section_doc, out=filecon, fragment=True)[0].split('\n')
        rinterface.baseenv['flush'](filecon)
        rinterface.baseenv['close'](filecon)
        with open(tempfilename) as fh:
            section_rows = fh.readlines()
    finally:
        os.unlink(tempfilename)
    return section_rows