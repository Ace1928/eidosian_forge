import os
import warnings
from collections import namedtuple
from importlib.util import find_spec
from io import BytesIO
import numpy
from rdkit import Chem
from rdkit import RDConfig
from rdkit import rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit.Chem.Draw.MolDrawing import MolDrawing
from rdkit.Chem.Draw.rdMolDraw2D import *
def MolDraw2DFromQPainter(qpainter, width=-1, height=-1, panelWidth=-1, panelHeight=-1):
    from rdkit.Chem.Draw import rdMolDraw2DQt
    if rdMolDraw2DQt.rdkitQtVersion.startswith('6'):
        from PyQt6.QtGui import QPainter
    else:
        from PyQt5.Qt import QPainter
    try:
        if rdMolDraw2DQt.rdkitQtVersion.startswith('6'):
            from PyQt6 import sip
        else:
            from PyQt5 import sip
    except ImportError:
        import sip
    if not isinstance(qpainter, QPainter):
        raise ValueError('argument must be a QPainter instance')
    if width <= 0:
        width = qpainter.viewport().width()
    if height <= 0:
        height = qpainter.viewport().height()
    ptr = sip.unwrapinstance(qpainter)
    d2d = rdMolDraw2DQt.MolDraw2DFromQPainter_(width, height, ptr, panelWidth, panelWidth)
    d2d._qptr = qpainter
    return d2d