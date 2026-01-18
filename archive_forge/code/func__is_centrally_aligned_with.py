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
def _is_centrally_aligned_with(self, other: LTComponent, tolerance: float=0) -> bool:
    """
        Whether the vertical center of `other` is within `tolerance`.
        """
    return abs((other.y0 + other.y1) / 2 - (self.y0 + self.y1) / 2) <= tolerance