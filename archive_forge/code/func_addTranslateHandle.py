import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def addTranslateHandle(self, pos, axes=None, item=None, name=None, index=None):
    """
        Add a new translation handle to the ROI. Dragging the handle will move 
        the entire ROI without changing its angle or shape. 
        
        Note that, by default, ROIs may be moved by dragging anywhere inside the
        ROI. However, for larger ROIs it may be desirable to disable this and
        instead provide one or more translation handles.
        
        =================== ====================================================
        **Arguments**
        pos                 (length-2 sequence) The position of the handle 
                            relative to the shape of the ROI. A value of (0,0)
                            indicates the origin, whereas (1, 1) indicates the
                            upper-right corner, regardless of the ROI's size.
        item                The Handle instance to add. If None, a new handle
                            will be created.
        name                The name of this handle (optional). Handles are 
                            identified by name when calling 
                            getLocalHandlePositions and getSceneHandlePositions.
        =================== ====================================================
        """
    pos = Point(pos)
    return self.addHandle({'name': name, 'type': 't', 'pos': pos, 'item': item}, index=index)