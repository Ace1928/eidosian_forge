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
class LTTextContainer(LTExpandableContainer[LTItemT], LTText):

    def __init__(self) -> None:
        LTText.__init__(self)
        LTExpandableContainer.__init__(self)
        return

    def get_text(self) -> str:
        return ''.join((cast(LTText, obj).get_text() for obj in self if isinstance(obj, LTText)))