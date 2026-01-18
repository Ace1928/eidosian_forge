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
class LTRect(LTCurve):
    """A rectangle.

    Could be used for framing another pictures or figures.
    """

    def __init__(self, linewidth: float, bbox: Rect, stroke: bool=False, fill: bool=False, evenodd: bool=False, stroking_color: Optional[Color]=None, non_stroking_color: Optional[Color]=None, original_path: Optional[List[PathSegment]]=None, dashing_style: Optional[Tuple[object, object]]=None) -> None:
        x0, y0, x1, y1 = bbox
        LTCurve.__init__(self, linewidth, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], stroke, fill, evenodd, stroking_color, non_stroking_color, original_path, dashing_style)