import copy
import functools
import math
import numpy
from rdkit import Chem
Set the molecule to be drawn.

    Parameters:
      hightlightAtoms -- list of atoms to highlight (default [])
      highlightMap -- dictionary of (atom, color) pairs (default None)

    Notes:
      - specifying centerIt will cause molTrans and drawingTrans to be ignored
    