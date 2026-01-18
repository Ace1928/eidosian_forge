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
def MolsToImage(mols, subImgSize=(200, 200), legends=None, **kwargs):
    """
  """
    from PIL import Image
    if legends is None:
        legends = [None] * len(mols)
    res = Image.new('RGBA', (subImgSize[0] * len(mols), subImgSize[1]))
    for i, mol in enumerate(mols):
        res.paste(MolToImage(mol, subImgSize, legend=legends[i], **kwargs), (i * subImgSize[0], 0))
    return res