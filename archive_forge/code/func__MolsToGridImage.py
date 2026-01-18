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
def _MolsToGridImage(mols, molsPerRow=3, subImgSize=(200, 200), legends=None, highlightAtomLists=None, highlightBondLists=None, drawOptions=None, returnPNG=False, **kwargs):
    """ returns a PIL Image of the grid
  """
    if legends is None:
        legends = [''] * len(mols)
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow:
        nRows += 1
    if not hasattr(rdMolDraw2D, 'MolDraw2DCairo'):
        from PIL import Image
        res = Image.new('RGBA', (molsPerRow * subImgSize[0], nRows * subImgSize[1]), (255, 255, 255, 0))
        for i, mol in enumerate(mols):
            row = i // molsPerRow
            col = i % molsPerRow
            highlights = None
            if highlightAtomLists and highlightAtomLists[i]:
                highlights = highlightAtomLists[i]
            if highlightBondLists and highlightBondLists[i]:
                kwargs['highlightBonds'] = highlightBondLists[i]
            if mol is not None:
                img = _moltoimg(mol, subImgSize, highlights, legends[i], **kwargs)
                res.paste(img, (col * subImgSize[0], row * subImgSize[1]))
    else:
        fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
        d2d = rdMolDraw2D.MolDraw2DCairo(fullSize[0], fullSize[1], subImgSize[0], subImgSize[1])
        if drawOptions is not None:
            d2d.SetDrawOptions(drawOptions)
        else:
            dops = d2d.drawOptions()
            for k, v in list(kwargs.items()):
                if hasattr(dops, k):
                    setattr(dops, k, v)
                    del kwargs[k]
        d2d.DrawMolecules(list(mols), legends=legends or None, highlightAtoms=highlightAtomLists, highlightBonds=highlightBondLists, **kwargs)
        d2d.FinishDrawing()
        if not returnPNG:
            res = _drawerToImage(d2d)
        else:
            res = d2d.GetDrawingText()
    return res