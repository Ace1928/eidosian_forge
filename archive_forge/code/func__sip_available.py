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
def _sip_available():
    try:
        from rdkit.Chem.Draw.rdMolDraw2DQt import rdkitQtVersion
    except ImportError:
        return False
    pyqt_pkg = f'PyQt{rdkitQtVersion[0]}'
    if find_spec(pyqt_pkg) and find_spec(f'{pyqt_pkg}.sip'):
        return True
    elif find_spec('sip'):
        return True
    return False