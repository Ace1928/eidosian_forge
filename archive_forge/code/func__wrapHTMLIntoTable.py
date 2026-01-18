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
def _wrapHTMLIntoTable(html):
    return InteractiveRenderer.injectHTMLFooterAfterTable(f'<div><table><tbody><tr><td style="width: {molSize[0]}px; ' + f'height: {molSize[1]}px; text-align: center;">' + html.replace(' scoped', '') + '</td></tr></tbody></table></div>')