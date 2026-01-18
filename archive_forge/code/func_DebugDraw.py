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
def DebugDraw(mol, size=(350, 350), drawer=None, asSVG=True, useBW=True, includeHLabels=True, addAtomIndices=True, addBondIndices=False):
    if drawer is None:
        if asSVG:
            drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        else:
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    if useBW:
        drawer.drawOptions().useBWAtomPalette()
    drawer.drawOptions().addAtomIndices = addAtomIndices
    drawer.drawOptions().addBondIndices = addBondIndices
    drawer.drawOptions().annotationFontScale = 0.75
    if includeHLabels:
        for atom in mol.GetAtoms():
            if atom.GetTotalNumHs():
                atom.SetProp('atomNote', f'H{atom.GetTotalNumHs()}')
    aromAtoms = [x.GetIdx() for x in mol.GetAtoms() if x.GetIsAromatic()]
    clrs = {x: (0.9, 0.9, 0.2) for x in aromAtoms}
    aromBonds = [x.GetIdx() for x in mol.GetBonds() if x.GetIsAromatic()]
    rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False, addChiralHs=False)
    drawer.drawOptions().prepareMolsBeforeDrawing = False
    drawer.DrawMolecule(mol, highlightAtoms=aromAtoms, highlightAtomColors=clrs, highlightBonds=aromBonds)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()