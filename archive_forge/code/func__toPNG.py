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
def _toPNG(mol):
    if hasattr(mol, '__sssAtoms'):
        highlightAtoms = mol.__sssAtoms
    else:
        highlightAtoms = []
    kekulize = kekulizeStructures
    return Draw._moltoimg(mol, molSize, highlightAtoms, '', returnPNG=True, kekulize=kekulize, drawOptions=drawOptions)