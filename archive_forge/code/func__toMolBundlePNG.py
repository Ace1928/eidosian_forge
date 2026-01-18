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
def _toMolBundlePNG(bundle):
    if _MolsToGridImageSaved is not None:
        fn = _MolsToGridImageSaved
    else:
        fn = Draw.MolsToGridImage
    return fn(bundle, subImgSize=molSize, drawOptions=drawOptions, useSVG=False, returnPNG=True)