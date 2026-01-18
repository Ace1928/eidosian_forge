import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def addRotateHandle(self, pos, center, item=None, name=None, index=None):
    """
        Add a new rotation handle to the ROI. Dragging a rotation handle allows 
        the user to change the angle of the ROI.
        
        =================== ====================================================
        **Arguments**
        pos                 (length-2 sequence) The position of the handle 
                            relative to the shape of the ROI. A value of (0,0)
                            indicates the origin, whereas (1, 1) indicates the
                            upper-right corner, regardless of the ROI's size.
        center              (length-2 sequence) The center point around which 
                            rotation takes place.
        item                The Handle instance to add. If None, a new handle
                            will be created.
        name                The name of this handle (optional). Handles are 
                            identified by name when calling 
                            getLocalHandlePositions and getSceneHandlePositions.
        =================== ====================================================
        """
    pos = Point(pos)
    center = Point(center)
    return self.addHandle({'name': name, 'type': 'r', 'center': center, 'pos': pos, 'item': item}, index=index)