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
def DrawMorganBits(tpls, **kwargs):
    envs = []
    for tpl in tpls:
        if len(tpl) == 4:
            mol, bitId, bitInfo, whichExample = tpl
        else:
            mol, bitId, bitInfo = tpl
            whichExample = 0
        atomId, radius = bitInfo[bitId][whichExample]
        envs.append((mol, atomId, radius))
    return DrawMorganEnvs(envs, **kwargs)