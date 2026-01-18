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
def DrawRDKitEnvs(envs, molsPerRow=3, subImgSize=(150, 150), baseRad=0.3, useSVG=True, aromaticColor=(0.9, 0.9, 0.2), extraColor=(0.9, 0.9, 0.9), nonAromaticColor=None, legends=None, drawOptions=None, **kwargs):
    submols = []
    highlightAtoms = []
    atomColors = []
    highlightBonds = []
    bondColors = []
    highlightRadii = []
    for mol, bondpath in envs:
        menv = _getRDKitEnv(mol, bondpath, baseRad, aromaticColor, extraColor, nonAromaticColor, **kwargs)
        submols.append(menv.submol)
        highlightAtoms.append(menv.highlightAtoms)
        atomColors.append(menv.atomColors)
        highlightBonds.append(menv.highlightBonds)
        bondColors.append(menv.bondColors)
        highlightRadii.append(menv.highlightRadii)
    if legends is None:
        legends = [''] * len(envs)
    nRows = len(envs) // molsPerRow
    if len(envs) % molsPerRow:
        nRows += 1
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    if useSVG:
        drawer = rdMolDraw2D.MolDraw2DSVG(fullSize[0], fullSize[1], subImgSize[0], subImgSize[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(fullSize[0], fullSize[1], subImgSize[0], subImgSize[1])
    if drawOptions is None:
        drawOptions = drawer.drawOptions()
    drawOptions.continuousHighlight = False
    drawOptions.includeMetadata = False
    drawer.SetDrawOptions(drawOptions)
    drawer.DrawMolecules(submols, legends=legends, highlightAtoms=highlightAtoms, highlightAtomColors=atomColors, highlightBonds=highlightBonds, highlightBondColors=bondColors, highlightAtomRadii=highlightRadii, **kwargs)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()