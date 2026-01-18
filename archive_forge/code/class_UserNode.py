import os, sys
from math import pi, cos, sin, sqrt, radians, floor
from reportlab.platypus import Flowable
from reportlab.rl_config import shapeChecking, verbose, defaultGraphicsFontName as _baseGFontName, _unset_, decimalSymbol
from reportlab.lib import logger
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.utils import isSeq, asBytes
from reportlab.lib.attrmap import *
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.fonts import tt2ps
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from . transform import *
class UserNode(_DrawTimeResizeable):
    """A simple template for creating a new node.  The user (Python
    programmer) may subclasses this.  provideNode() must be defined to
    provide a Shape primitive when called by a renderer.  It does
    NOT inherit from Shape, as the renderer always replaces it, and
    your own classes can safely inherit from it without getting
    lots of unintended behaviour."""

    def provideNode(self):
        """Override this to create your own node. This lets widgets be
        added to drawings; they must create a shape (typically a group)
        so that the renderer can draw the custom node."""
        raise NotImplementedError('this method must be redefined by the user/programmer')