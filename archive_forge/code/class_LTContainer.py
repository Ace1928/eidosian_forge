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
class LTContainer(LTComponent, Generic[LTItemT]):
    """Object that can be extended and analyzed"""

    def __init__(self, bbox: Rect) -> None:
        LTComponent.__init__(self, bbox)
        self._objs: List[LTItemT] = []
        return

    def __iter__(self) -> Iterator[LTItemT]:
        return iter(self._objs)

    def __len__(self) -> int:
        return len(self._objs)

    def add(self, obj: LTItemT) -> None:
        self._objs.append(obj)
        return

    def extend(self, objs: Iterable[LTItemT]) -> None:
        for obj in objs:
            self.add(obj)
        return

    def analyze(self, laparams: LAParams) -> None:
        for obj in self._objs:
            obj.analyze(laparams)
        return