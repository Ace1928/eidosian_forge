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
def MolsMatrixToGridImage(molsMatrix, subImgSize=(200, 200), legendsMatrix=None, highlightAtomListsMatrix=None, highlightBondListsMatrix=None, useSVG=False, returnPNG=False, **kwargs):
    """Creates a mol grid image from a nested data structure (where each data substructure represents a row),
  padding rows as needed so all rows are the length of the longest row
          ARGUMENTS:

        - molsMatrix: A two-deep nested data structure of RDKit molecules to draw,
         iterable of iterables (for example list of lists) of RDKit molecules

        - subImgSize: The size of a cell in the drawing; passed through to MolsToGridImage (default (200, 200))

        - legendsMatrix: A two-deep nested data structure of strings to label molecules with,
         iterable of iterables (for example list of lists) of strings (default None)

        - highlightAtomListsMatrix: A three-deep nested data structure of integers of atoms to highlight,
         iterable of iterables (for example list of lists) of integers (default None)

        - highlightBondListsMatrix: A three-deep nested data structure of integers of bonds to highlight,
         iterable of iterables (for example list of lists) of integers (default None)

        - useSVG: Whether to return an SVG (if true) or PNG (if false);
         passed through to MolsToGridImage (default false)

        - returnPNG: Whether to return PNG data (if true) or a PIL object for a PNG image file (if false);
         has no effect if useSVG is true; passed through to MolsToGridImage (default false)

        - kwargs: Any other keyword arguments are passed to MolsToGridImage

      NOTES:

            To include a blank cell in the middle of a row, supply None for that entry in molsMatrix.
            You do not need to do that for empty cells at the end of a row; 
            this function will automatically pad rows so that all rows are the same length.
            
            This function is useful when each row has some meaning,
            for example the generation in a mass spectrometry fragmentation tree--refer to 
            example at https://en.wikipedia.org/wiki/Fragmentation_(mass_spectrometry).
            If you want to display a set molecules where each row does not have any specific meaning,
            use MolsToGridImage instead.

            This function nests data structures one additional level beyond the analogous function MolsToGridImage
            (in which the molecules and legends are non-nested lists, 
            and the highlight parameters are two-deep nested lists) 

      RETURNS:

        A grid of molecular images in one of these formats:
        
        - useSVG=False and returnPNG=False (default): A PIL object for a PNG image file

        - useSVG=False and returnPNG=True: PNG data

        - useSVG=True: An SVG string

      EXAMPLES:

        from rdkit import Chem
        from rdkit.Chem.Draw import MolsMatrixToGridImage, rdMolDraw2D
        FCl = Chem.MolFromSmiles("FCl")
        molsMatrix = [[FCl, FCl], [FCl, None, FCl]]

        # Minimal example: Only molsMatrix is supplied,
        # result will be a drawing containing (where each row contains molecules):
        # F-Cl    F-Cl
        # F-Cl            F-Cl
        img = MolsMatrixToGridImage(molsMatrix)
        img.save("MolsMatrixToGridImageMinimal.png")
        # img is a PIL object for a PNG image file like:
        # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=600x200 at 0x1648CC390>
        # Drawing will be saved as PNG file MolsMatrixToGridImageMinimal.png

        # Exhaustive example: All parameters are supplied,
        # result will be a drawing containing (where each row of molecules is followed by a row of legends):
        # 1 F-Cl 0              1 F-Cl 0
        # no highlighting       bond highlighted         
        # 1 F-Cl 0                                  1 F-Cl 0
        # sodium highlighted                        chloride and bond highlighted
        legendsMatrix = [["no highlighting", "bond highlighted"], 
        ["F highlighted", "", "Cl and bond highlighted"]]
        highlightAtomListsMatrix = [[[],[]], [[0], None, [1]]]
        highlightBondListsMatrix = [[[],[0]], [[], None, [0]]]

        dopts = rdMolDraw2D.MolDrawOptions()
        dopts.addAtomIndices = True

        img_binary = MolsMatrixToGridImage(molsMatrix=molsMatrix, subImgSize=(300, 400), 
        legendsMatrix=legendsMatrix, highlightAtomListsMatrix=highlightAtomListsMatrix, 
        highlightBondListsMatrix=highlightBondListsMatrix, useSVG=False, returnPNG=True, drawOptions=dopts)
        print(img_binary[:20])
        # Prints a binary string: b'\x89PNG\r
\x1a
\x00\x00\x00\rIHDR\x00\x00\x03\x84'
  """
    mols, molsPerRow, legends, highlightAtomLists, highlightBondLists = _MolsNestedToLinear(molsMatrix, legendsMatrix, highlightAtomListsMatrix, highlightBondListsMatrix)
    return MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=subImgSize, legends=legends, highlightAtomLists=highlightAtomLists, highlightBondLists=highlightBondLists, **kwargs)