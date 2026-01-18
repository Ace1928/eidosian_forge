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
class LTTextBox(LTTextContainer[LTTextLine]):
    """Represents a group of text chunks in a rectangular area.

    Note that this box is created by geometric analysis and does not
    necessarily represents a logical boundary of the text. It contains a list
    of LTTextLine objects.
    """

    def __init__(self) -> None:
        LTTextContainer.__init__(self)
        self.index: int = -1
        return

    def __repr__(self) -> str:
        return '<%s(%s) %s %r>' % (self.__class__.__name__, self.index, bbox2str(self.bbox), self.get_text())

    def get_writing_mode(self) -> str:
        raise NotImplementedError