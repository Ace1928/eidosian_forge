from math import hypot
from .. import functions as fn
from ..Qt import QtGui, QtWidgets

        Changes the appearance of the arrow.
        All arguments are optional:
        
        ======================  =================================================
        **Keyword Arguments:**
        angle                   Orientation of the arrow in degrees. Default is
                                0; arrow pointing to the left.
        headLen                 Length of the arrow head, from tip to base.
                                default=20
        headWidth               Width of the arrow head at its base. If
                                headWidth is specified, it overrides tipAngle.
        tipAngle                Angle of the tip of the arrow in degrees. Smaller
                                values make a 'sharper' arrow. default=25
        baseAngle               Angle of the base of the arrow head. Default is
                                0, which means that the base of the arrow head
                                is perpendicular to the arrow tail.
        tailLen                 Length of the arrow tail, measured from the base
                                of the arrow head to the end of the tail. If
                                this value is None, no tail will be drawn.
                                default=None
        tailWidth               Width of the tail. default=3
        pen                     The pen used to draw the outline of the arrow.
        brush                   The brush used to fill the arrow.
        pxMode                  If True, then the arrow is drawn as a fixed size
                                regardless of the scale of its parents (including
                                the ViewBox zoom level). 
        ======================  =================================================
        