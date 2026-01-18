import heapq
import logging
from typing import (
from .pdfcolor import PDFColorSpace
from .pdffont import PDFFont
from .pdfinterp import Color
from .pdfinterp import PDFGraphicState
from .pdftypes import PDFStream
from .utils import INF, PathSegment
from .utils import LTComponentT
from .utils import Matrix
from .utils import Plane
from .utils import Point
from .utils import Rect
from .utils import apply_matrix_pt
from .utils import bbox2str
from .utils import fsplit
from .utils import get_bound
from .utils import matrix2str
from .utils import uniq
class LTLine(LTCurve):
    """A single straight line.

    Could be used for separating text or figures.
    """

    def __init__(self, linewidth: float, p0: Point, p1: Point, stroke: bool=False, fill: bool=False, evenodd: bool=False, stroking_color: Optional[Color]=None, non_stroking_color: Optional[Color]=None, original_path: Optional[List[PathSegment]]=None, dashing_style: Optional[Tuple[object, object]]=None) -> None:
        LTCurve.__init__(self, linewidth, [p0, p1], stroke, fill, evenodd, stroking_color, non_stroking_color, original_path, dashing_style)