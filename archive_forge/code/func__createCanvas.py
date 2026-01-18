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
def _createCanvas(size):
    useAGG, useCairo, Canvas = _getCanvas()
    if useAGG or useCairo:
        from PIL import Image
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        canvas = Canvas(img)
    else:
        from rdkit.Chem.Draw.spingCanvas import Canvas
        canvas = Canvas(size=size, name='MolToImageFile')
        img = canvas._image
    return (img, canvas)