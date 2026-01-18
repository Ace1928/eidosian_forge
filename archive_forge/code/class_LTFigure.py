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
class LTFigure(LTLayoutContainer):
    """Represents an area used by PDF Form objects.

    PDF Forms can be used to present figures or pictures by embedding yet
    another PDF document within a page. Note that LTFigure objects can appear
    recursively.
    """

    def __init__(self, name: str, bbox: Rect, matrix: Matrix) -> None:
        self.name = name
        self.matrix = matrix
        x, y, w, h = bbox
        bounds = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
        bbox = get_bound((apply_matrix_pt(matrix, (p, q)) for p, q in bounds))
        LTLayoutContainer.__init__(self, bbox)
        return

    def __repr__(self) -> str:
        return '<%s(%s) %s matrix=%s>' % (self.__class__.__name__, self.name, bbox2str(self.bbox), matrix2str(self.matrix))

    def analyze(self, laparams: LAParams) -> None:
        if not laparams.all_texts:
            return
        LTLayoutContainer.analyze(self, laparams)
        return