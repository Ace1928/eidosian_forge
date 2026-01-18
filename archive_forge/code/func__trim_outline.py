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
def _trim_outline(self, pdf: PdfReader, outline: OutlineType, pages: Union[Tuple[int, int], Tuple[int, int, int], List[int]]) -> OutlineType:
    """
        Remove outline item entries that are not a part of the specified page set.

        Args:
            pdf:
            outline:
            pages:

        Returns:
            An outline type
        """
    new_outline = []
    prev_header_added = True
    lst = pages if isinstance(pages, list) else list(range(*pages))
    for i, outline_item in enumerate(outline):
        if isinstance(outline_item, list):
            sub = self._trim_outline(pdf, outline_item, lst)
            if sub:
                if not prev_header_added:
                    new_outline.append(outline[i - 1])
                new_outline.append(sub)
        else:
            prev_header_added = False
            for j in lst:
                if outline_item['/Page'] is None:
                    continue
                if pdf.pages[j].get_object() == outline_item['/Page'].get_object():
                    outline_item[NameObject('/Page')] = outline_item['/Page'].get_object()
                    new_outline.append(outline_item)
                    prev_header_added = True
                    break
    return new_outline