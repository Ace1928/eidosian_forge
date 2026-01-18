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
def _DrawBits(fn, *args, **kwargs):
    if 'useSVG' not in kwargs:
        kwargs['useSVG'] = ipython_useSVG
    res = fn(*args, **kwargs)
    if kwargs['useSVG']:
        return SVG(res)
    sio = BytesIO(res)
    return Image.open(sio)