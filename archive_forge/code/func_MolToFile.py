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
def MolToFile(mol, filename, size=(300, 300), kekulize=True, wedgeBonds=True, imageType=None, fitImage=False, options=None, **kwargs):
    """ Generates a drawing of a molecule and writes it to a file
  """
    if not filename:
        raise ValueError('no filename provided')
    if not mol:
        raise ValueError('Null molecule provided')
    if imageType is None:
        imageType = os.path.splitext(filename)[1][1:]
    if imageType not in ('svg', 'png'):
        _legacyMolToFile(mol, filename, size, kekulize, wedgeBonds, imageType, fitImage, options, **kwargs)
    if type(options) == DrawingOptions:
        warnings.warn('legacy DrawingOptions not translated for new drawing code, please update manually', DeprecationWarning)
        options = None
    if imageType == 'png':
        drawfn = _moltoimg
        mode = 'b'
    elif imageType == 'svg':
        drawfn = _moltoSVG
        mode = 't'
    else:
        raise ValueError('unsupported output format')
    data = drawfn(mol, size, kwargs.get('highlightAtoms', []), kwargs.get('legend', ''), highlightBonds=kwargs.get('highlightBonds', []), drawOptions=options, kekulize=kekulize, wedgeBonds=wedgeBonds, returnPNG=True)
    with open(filename, 'w+' + mode) as outf:
        outf.write(data)
        outf.close()