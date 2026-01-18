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
def ReactionToImage(rxn, subImgSize=(200, 200), useSVG=False, drawOptions=None, returnPNG=False, **kwargs):
    if not useSVG and (not hasattr(rdMolDraw2D, 'MolDraw2DCairo')):
        return _legacyReactionToImage(rxn, subImgSize=subImgSize, **kwargs)
    else:
        width = subImgSize[0] * (rxn.GetNumReactantTemplates() + rxn.GetNumProductTemplates() + 1)
        if useSVG:
            d = rdMolDraw2D.MolDraw2DSVG(width, subImgSize[1])
        else:
            d = rdMolDraw2D.MolDraw2DCairo(width, subImgSize[1])
        if drawOptions is not None:
            d.SetDrawOptions(drawOptions)
        d.DrawReaction(rxn, **kwargs)
        d.FinishDrawing()
        if useSVG or returnPNG:
            return d.GetDrawingText()
        else:
            return _drawerToImage(d)