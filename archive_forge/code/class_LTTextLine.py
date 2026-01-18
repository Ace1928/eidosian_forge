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
class LTTextLine(LTTextContainer[TextLineElement]):
    """Contains a list of LTChar objects that represent a single text line.

    The characters are aligned either horizontally or vertically, depending on
    the text's writing mode.
    """

    def __init__(self, word_margin: float) -> None:
        super().__init__()
        self.word_margin = word_margin
        return

    def __repr__(self) -> str:
        return '<%s %s %r>' % (self.__class__.__name__, bbox2str(self.bbox), self.get_text())

    def analyze(self, laparams: LAParams) -> None:
        for obj in self._objs:
            obj.analyze(laparams)
        LTContainer.add(self, LTAnno('\n'))
        return

    def find_neighbors(self, plane: Plane[LTComponentT], ratio: float) -> List['LTTextLine']:
        raise NotImplementedError

    def is_empty(self) -> bool:
        return super().is_empty() or self.get_text().isspace()