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
def _toJSON(mol):
    """For IPython notebook, renders 3D webGL objects."""
    if not ipython_3d or not mol.GetNumConformers():
        return None
    conf = mol.GetConformer()
    if not conf.Is3D():
        return None
    res = drawMol3D(mol)
    if hasattr(res, 'data'):
        return res.data
    return ''