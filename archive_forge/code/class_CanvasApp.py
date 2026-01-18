import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
class CanvasApp(wxApp):
    """The wxApp that runs canvas.  Initializes windows, and handles redrawing"""

    def OnInit(self):
        return 1