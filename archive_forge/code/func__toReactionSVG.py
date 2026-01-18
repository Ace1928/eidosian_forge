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
def _toReactionSVG(rxn):
    if not ipython_useSVG:
        return None
    rc = copy.deepcopy(rxn)
    return Draw.ReactionToImage(rc, subImgSize=(int(molSize[0] / 3), molSize[1]), useSVG=True, highlightByReactant=highlightByReactant, drawOptions=drawOptions)