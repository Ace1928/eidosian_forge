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
def DisableSubstructMatchRendering():
    if hasattr(rdchem.Mol, '__GetSubstructMatch'):
        rdchem.Mol.GetSubstructMatch = rdchem.Mol.__GetSubstructMatch
        del rdchem.Mol.__GetSubstructMatch
    if hasattr(rdchem.Mol, '__GetSubstructMatches'):
        rdchem.Mol.GetSubstructMatches = rdchem.Mol.__GetSubstructMatches
        del rdchem.Mol.__GetSubstructMatches