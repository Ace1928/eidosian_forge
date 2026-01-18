import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def _getArrayRegionForArbitraryShape(self, data, img, axes=(0, 1), returnMappedCoords=False, **kwds):
    """
        Return the result of :meth:`~pyqtgraph.ROI.getArrayRegion`, masked by
        the shape of the ROI. Values outside the ROI shape are set to 0.

        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the
        arguments.
        """
    if returnMappedCoords:
        sliced, mappedCoords = ROI.getArrayRegion(self, data, img, axes, returnMappedCoords, fromBoundingRect=True, **kwds)
    else:
        sliced = ROI.getArrayRegion(self, data, img, axes, returnMappedCoords, fromBoundingRect=True, **kwds)
    if img.axisOrder == 'col-major':
        mask = self.renderShapeMask(sliced.shape[axes[0]], sliced.shape[axes[1]])
    else:
        mask = self.renderShapeMask(sliced.shape[axes[1]], sliced.shape[axes[0]])
        mask = mask.T
    shape = [1] * data.ndim
    shape[axes[0]] = sliced.shape[axes[0]]
    shape[axes[1]] = sliced.shape[axes[1]]
    mask = mask.reshape(shape)
    if returnMappedCoords:
        return (sliced * mask, mappedCoords)
    else:
        return sliced * mask