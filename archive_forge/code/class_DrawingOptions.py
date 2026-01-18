import copy
import functools
import math
import numpy
from rdkit import Chem
class DrawingOptions(object):
    dotsPerAngstrom = 30
    useFraction = 0.85
    atomLabelFontFace = 'sans'
    atomLabelFontSize = 12
    atomLabelMinFontSize = 7
    atomLabelDeuteriumTritium = False
    bondLineWidth = 1.2
    dblBondOffset = 0.25
    dblBondLengthFrac = 0.8
    defaultColor = (1, 0, 0)
    selectColor = (1, 0, 0)
    bgColor = (1, 1, 1)
    colorBonds = True
    noCarbonSymbols = True
    includeAtomNumbers = False
    atomNumberOffset = 0
    radicalSymbol = u'∙'
    dash = (4, 4)
    wedgeDashedBonds = True
    showUnknownDoubleBonds = True
    coordScale = 1.0
    elemDict = {1: (0.55, 0.55, 0.55), 7: (0, 0, 1), 8: (1, 0, 0), 9: (0.2, 0.8, 0.8), 15: (1, 0.5, 0), 16: (0.8, 0.8, 0), 17: (0, 0.8, 0), 35: (0.5, 0.3, 0.1), 53: (0.63, 0.12, 0.94), 0: (0.5, 0.5, 0.5)}