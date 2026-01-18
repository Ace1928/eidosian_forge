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
class LTCurve(LTComponent):
    """
    A generic Bezier curve

    The parameter `original_path` contains the original
    pathing information from the pdf (e.g. for reconstructing Bezier Curves).

    `dashing_style` contains the Dashing information if any.
    """

    def __init__(self, linewidth: float, pts: List[Point], stroke: bool=False, fill: bool=False, evenodd: bool=False, stroking_color: Optional[Color]=None, non_stroking_color: Optional[Color]=None, original_path: Optional[List[PathSegment]]=None, dashing_style: Optional[Tuple[object, object]]=None) -> None:
        LTComponent.__init__(self, get_bound(pts))
        self.pts = pts
        self.linewidth = linewidth
        self.stroke = stroke
        self.fill = fill
        self.evenodd = evenodd
        self.stroking_color = stroking_color
        self.non_stroking_color = non_stroking_color
        self.original_path = original_path
        self.dashing_style = dashing_style

    def get_pts(self) -> str:
        return ','.join(('%.3f,%.3f' % p for p in self.pts))