import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def addScaleRotateHandle(self, pos, center, item=None, name=None, index=None):
    """
        Add a new scale+rotation handle to the ROI. When dragging a handle of 
        this type, the user can simultaneously rotate the ROI around an 
        arbitrary center point as well as scale the ROI by dragging the handle
        toward or away from the center point.
        
        =================== ====================================================
        **Arguments**
        pos                 (length-2 sequence) The position of the handle 
                            relative to the shape of the ROI. A value of (0,0)
                            indicates the origin, whereas (1, 1) indicates the
                            upper-right corner, regardless of the ROI's size.
        center              (length-2 sequence) The center point around which 
                            scaling and rotation take place.
        item                The Handle instance to add. If None, a new handle
                            will be created.
        name                The name of this handle (optional). Handles are 
                            identified by name when calling 
                            getLocalHandlePositions and getSceneHandlePositions.
        =================== ====================================================
        """
    pos = Point(pos)
    center = Point(center)
    if pos[0] == center[0] and pos[1] == center[1]:
        raise Exception('Scale/rotate handles cannot be at their center point.')
    return self.addHandle({'name': name, 'type': 'sr', 'center': center, 'pos': pos, 'item': item}, index=index)