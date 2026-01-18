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
def _write_outline_item_on_page(self, outline_item: Union[OutlineItem, Destination], page: _MergedPage) -> None:
    oi_type = cast(str, outline_item['/Type'])
    args = [NumberObject(page.id), NameObject(oi_type)]
    fit2arg_keys: Dict[str, Tuple[str, ...]] = {TypFitArguments.FIT_H: (TypArguments.TOP,), TypFitArguments.FIT_BH: (TypArguments.TOP,), TypFitArguments.FIT_V: (TypArguments.LEFT,), TypFitArguments.FIT_BV: (TypArguments.LEFT,), TypFitArguments.XYZ: (TypArguments.LEFT, TypArguments.TOP, '/Zoom'), TypFitArguments.FIT_R: (TypArguments.LEFT, TypArguments.BOTTOM, TypArguments.RIGHT, TypArguments.TOP)}
    for arg_key in fit2arg_keys.get(oi_type, ()):
        if arg_key in outline_item and (not isinstance(outline_item[arg_key], NullObject)):
            args.append(FloatObject(outline_item[arg_key]))
        else:
            args.append(FloatObject(0))
        del outline_item[arg_key]
    outline_item[NameObject('/A')] = DictionaryObject({NameObject(GoToActionArguments.S): NameObject('/GoTo'), NameObject(GoToActionArguments.D): ArrayObject(args)})