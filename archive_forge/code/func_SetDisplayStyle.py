import os
import sys
import tempfile
from rdkit import Chem
def SetDisplayStyle(self, obj, style=''):
    """ change the display style of the specified object """
    self.server.do('hide everything,%s' % (obj,))
    if style:
        self.server.do('show %s,%s' % (style, obj))