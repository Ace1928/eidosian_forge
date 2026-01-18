import copy
import functools
import math
import numpy
from rdkit import Chem
def _drawWedgedBond(self, bond, pos, nbrPos, width=None, color=None, dash=None):
    width = width or self.drawingOptions.bondLineWidth
    color = color or self.drawingOptions.defaultColor
    _, offsetX, offsetY = self._getBondOffset(pos, nbrPos)
    offsetX *= 0.75
    offsetY *= 0.75
    poly = ((pos[0], pos[1]), (nbrPos[0] + offsetX, nbrPos[1] + offsetY), (nbrPos[0] - offsetX, nbrPos[1] - offsetY))
    if not dash:
        self.canvas.addCanvasPolygon(poly, color=color)
    elif self.drawingOptions.wedgeDashedBonds and self.canvas.addCanvasDashedWedge:
        self.canvas.addCanvasDashedWedge(poly[0], poly[1], poly[2], color=color)
    else:
        self.canvas.addCanvasLine(pos, nbrPos, linewidth=width * 2, color=color, dashes=dash)