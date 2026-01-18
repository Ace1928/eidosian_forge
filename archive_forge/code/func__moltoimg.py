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
def _moltoimg(mol, sz, highlights, legend, returnPNG=False, drawOptions=None, **kwargs):
    try:
        with rdBase.BlockLogs():
            mol.GetAtomWithIdx(0).GetExplicitValence()
    except RuntimeError:
        mol.UpdatePropertyCache(False)
    kekulize = shouldKekulize(mol, kwargs.get('kekulize', True))
    wedge = kwargs.get('wedgeBonds', True)
    if not drawOptions or drawOptions.prepareMolsBeforeDrawing:
        try:
            with rdBase.BlockLogs():
                mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=kekulize, wedgeBonds=wedge)
        except ValueError:
            mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False, wedgeBonds=wedge)
    if not hasattr(rdMolDraw2D, 'MolDraw2DCairo'):
        img = MolToImage(mol, sz, legend=legend, highlightAtoms=highlights, **kwargs)
        if returnPNG:
            bio = BytesIO()
            img.save(bio, format='PNG')
            img = bio.getvalue()
    else:
        d2d = rdMolDraw2D.MolDraw2DCairo(sz[0], sz[1])
        if drawOptions is not None:
            d2d.SetDrawOptions(drawOptions)
        if 'highlightColor' in kwargs and kwargs['highlightColor']:
            d2d.drawOptions().setHighlightColour(kwargs['highlightColor'])
        d2d.drawOptions().prepareMolsBeforeDrawing = False
        bondHighlights = kwargs.get('highlightBonds', None)
        if bondHighlights is not None:
            d2d.DrawMolecule(mol, legend=legend or '', highlightAtoms=highlights or [], highlightBonds=bondHighlights)
        else:
            d2d.DrawMolecule(mol, legend=legend or '', highlightAtoms=highlights or [])
        d2d.FinishDrawing()
        if returnPNG:
            img = d2d.GetDrawingText()
        else:
            img = _drawerToImage(d2d)
    return img