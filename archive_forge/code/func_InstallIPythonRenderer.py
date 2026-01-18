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
def InstallIPythonRenderer():
    global _MolsToGridImageSaved, _DrawRDKitBitSaved, _DrawRDKitBitsSaved, _DrawMorganBitSaved, _DrawMorganBitsSaved
    global _rendererInstalled
    if _rendererInstalled:
        return
    rdchem.Mol._repr_png_ = _toPNG
    rdchem.Mol._repr_svg_ = _toSVG
    _methodsToDelete.append((rdchem.Mol, '_repr_png_'))
    _methodsToDelete.append((rdchem.Mol, '_repr_svg_'))
    rdchem.Mol._repr_html_ = _toHTML
    _methodsToDelete.append((rdchem.Mol, '_repr_html_'))
    rdChemReactions.ChemicalReaction._repr_png_ = _toReactionPNG
    rdChemReactions.ChemicalReaction._repr_svg_ = _toReactionSVG
    _methodsToDelete.append((rdChemReactions.ChemicalReaction, '_repr_png_'))
    _methodsToDelete.append((rdChemReactions.ChemicalReaction, '_repr_svg_'))
    rdchem.MolBundle._repr_png_ = _toMolBundlePNG
    rdchem.MolBundle._repr_svg_ = _toMolBundleSVG
    _methodsToDelete.append((rdchem.MolBundle, '_repr_png_'))
    _methodsToDelete.append((rdchem.MolBundle, '_repr_svg_'))
    EnableSubstructMatchRendering()
    Image.Image._repr_png_ = display_pil_image
    _methodsToDelete.append((Image.Image, '_repr_png_'))
    _MolsToGridImageSaved = Draw.MolsToGridImage
    Draw.MolsToGridImage = ShowMols
    _DrawRDKitBitSaved = Draw.DrawRDKitBit
    Draw.DrawRDKitBit = DrawRDKitBit
    _DrawRDKitBitsSaved = Draw.DrawRDKitBits
    Draw.DrawRDKitBits = DrawRDKitBits
    _DrawMorganBitSaved = Draw.DrawMorganBit
    Draw.DrawMorganBit = DrawMorganBit
    _DrawMorganBitsSaved = Draw.DrawMorganBits
    Draw.DrawMorganBits = DrawMorganBits
    rdchem.Mol.__DebugMol = rdchem.Mol.Debug
    rdchem.Mol.Debug = lambda self, useStdout=False: self.__DebugMol(useStdout=useStdout)
    _rendererInstalled = True