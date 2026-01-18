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
def _GetSubstructMatches(mol, query, *args, **kwargs):
    res = mol.__GetSubstructMatches(query, *args, **kwargs)
    mol.__sssAtoms = []
    if highlightSubstructs:
        for entry in res:
            mol.__sssAtoms.extend(list(entry))
    return res