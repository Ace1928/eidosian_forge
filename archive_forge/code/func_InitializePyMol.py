import os
import sys
import tempfile
from rdkit import Chem
def InitializePyMol(self):
    """ does some initializations to set up PyMol according to our
    tastes

    """
    self.server.do('set valence,1')
    self.server.do('set stick_rad,0.15')
    self.server.do('set mouse_selection_mode,0')
    self.server.do('set line_width,2')
    self.server.do('set selection_width,10')
    self.server.do('set auto_zoom,0')