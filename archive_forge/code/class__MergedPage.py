from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import (
from ._encryption import Encryption
from ._page import PageObject
from ._reader import PdfReader
from ._utils import (
from ._writer import PdfWriter
from .constants import GoToActionArguments, TypArguments, TypFitArguments
from .constants import PagesAttributes as PA
from .generic import (
from .pagerange import PageRange, PageRangeSpec
from .types import LayoutType, OutlineType, PagemodeType
class _MergedPage:
    """Collect necessary information on each page that is being merged."""

    def __init__(self, pagedata: PageObject, src: PdfReader, id: int) -> None:
        self.src = src
        self.pagedata = pagedata
        self.out_pagedata = None
        self.id = id