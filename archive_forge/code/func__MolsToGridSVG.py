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
def _MolsToGridSVG(mols, molsPerRow=3, subImgSize=(200, 200), legends=None, highlightAtomLists=None, highlightBondLists=None, drawOptions=None, **kwargs):
    """ returns an SVG of the grid
  """
    if legends is None:
        legends = [''] * len(mols)
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow:
        nRows += 1
    blocks = [''] * (nRows * molsPerRow)
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    d2d = rdMolDraw2D.MolDraw2DSVG(fullSize[0], fullSize[1], subImgSize[0], subImgSize[1])
    if drawOptions is not None:
        d2d.SetDrawOptions(drawOptions)
    else:
        dops = d2d.drawOptions()
        for k, v in list(kwargs.items()):
            if hasattr(dops, k):
                setattr(dops, k, v)
                del kwargs[k]
    d2d.DrawMolecules(list(mols), legends=legends or None, highlightAtoms=highlightAtomLists or [], highlightBonds=highlightBondLists or [], **kwargs)
    d2d.FinishDrawing()
    res = d2d.GetDrawingText()
    return res