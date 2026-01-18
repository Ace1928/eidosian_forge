import base64
import copy
import html
import warnings
from io import BytesIO
import IPython
from IPython.display import HTML, SVG
from rdkit import Chem
from rdkit.Chem import Draw, rdchem, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from . import InteractiveRenderer
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from IPython import display
def addMolToView(mol, view, confId=-1, drawAs=None):
    if mol.GetNumAtoms() >= 999 or drawAs == 'cartoon':
        pdb = Chem.MolToPDBBlock(mol, flavor=32 | 16)
        view.addModel(pdb, 'pdb')
    else:
        mb = Chem.MolToMolBlock(mol, confId=confId)
        view.addModel(mb, 'sdf')
    if drawAs is None:
        drawAs = drawing_type_3d
    view.setStyle({drawAs: {}})