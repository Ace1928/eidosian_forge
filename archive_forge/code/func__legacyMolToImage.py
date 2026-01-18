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
def _legacyMolToImage(mol, size, kekulize, wedgeBonds, fitImage, options, canvas, **kwargs):
    """Returns a PIL image containing a drawing of the molecule using the legacy drawing code

      ARGUMENTS:

        - kekulize: run kekulization routine on input `mol` (default True)

        - size: final image size, in pixel (default (300,300))

        - wedgeBonds: draw wedge (stereo) bonds (default True)

        - highlightAtoms: list of atoms to highlight (default [])

        - highlightMap: dictionary of (atom, color) pairs (default None)

        - highlightBonds: list of bonds to highlight (default [])

        - highlightColor: RGB color as tuple (default [1, 0, 0])

      NOTE:

            use 'matplotlib.colors.to_rgb()' to convert string and
            HTML color codes into the RGB tuple representation, eg.

              from matplotlib.colors import ColorConverter
              img = Draw.MolToImage(m, highlightAtoms=[1,2], highlightColor=ColorConverter().to_rgb('aqua'))
              img.save("molecule.png")

      RETURNS:

        a PIL Image object
  """
    if not mol:
        raise ValueError('Null molecule provided')
    if canvas is None:
        img, canvas = _createCanvas(size)
    else:
        img = None
    options = options or DrawingOptions()
    if fitImage:
        options.dotsPerAngstrom = int(min(size) / 10)
    options.wedgeDashedBonds = wedgeBonds
    if 'highlightColor' in kwargs:
        color = kwargs.pop('highlightColor', (1, 0, 0))
        options.selectColor = color
    drawer = MolDrawing(canvas=canvas, drawingOptions=options)
    if kekulize:
        from rdkit import Chem
        mol = Chem.Mol(mol.ToBinary())
        Chem.Kekulize(mol)
    if not mol.GetNumConformers():
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
    if 'legend' in kwargs:
        legend = kwargs['legend']
        del kwargs['legend']
    else:
        legend = ''
    drawer.AddMol(mol, **kwargs)
    if legend:
        from rdkit.Chem.Draw.MolDrawing import Font
        bbox = drawer.boundingBoxes[mol]
        pos = (size[0] / 2, int(0.94 * size[1]), 0)
        font = Font(face='sans', size=12)
        canvas.addCanvasText(legend, pos, font)
    if kwargs.get('returnCanvas', False):
        return (img, canvas, drawer)
    else:
        canvas.flush()
        return img