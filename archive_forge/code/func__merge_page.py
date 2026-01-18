import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import (
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfCommonDocProtocol
from ._text_extraction import (
from ._utils import (
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Resources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import (
def _merge_page(self, page2: 'PageObject', page2transformation: Optional[Callable[[Any], ContentStream]]=None, ctm: Optional[CompressedTransformationMatrix]=None, over: bool=True, expand: bool=False) -> None:
    try:
        assert isinstance(self.indirect_reference, IndirectObject)
        if hasattr(self.indirect_reference.pdf, '_add_object'):
            return self._merge_page_writer(page2, page2transformation, ctm, over, expand)
    except (AssertionError, AttributeError):
        pass
    new_resources = DictionaryObject()
    rename = {}
    try:
        original_resources = cast(DictionaryObject, self[PG.RESOURCES].get_object())
    except KeyError:
        original_resources = DictionaryObject()
    try:
        page2resources = cast(DictionaryObject, page2[PG.RESOURCES].get_object())
    except KeyError:
        page2resources = DictionaryObject()
    new_annots = ArrayObject()
    for page in (self, page2):
        if PG.ANNOTS in page:
            annots = page[PG.ANNOTS]
            if isinstance(annots, ArrayObject):
                new_annots.extend(annots)
    for res in (RES.EXT_G_STATE, RES.FONT, RES.XOBJECT, RES.COLOR_SPACE, RES.PATTERN, RES.SHADING, RES.PROPERTIES):
        new, newrename = self._merge_resources(original_resources, page2resources, res)
        if new:
            new_resources[NameObject(res)] = new
            rename.update(newrename)
    new_resources[NameObject(RES.PROC_SET)] = ArrayObject(sorted(set(original_resources.get(RES.PROC_SET, ArrayObject()).get_object()).union(set(page2resources.get(RES.PROC_SET, ArrayObject()).get_object()))))
    new_content_array = ArrayObject()
    original_content = self.get_contents()
    if original_content is not None:
        original_content.isolate_graphics_state()
        new_content_array.append(original_content)
    page2content = page2.get_contents()
    if page2content is not None:
        rect = getattr(page2, MERGE_CROP_BOX)
        page2content.operations.insert(0, (map(FloatObject, [rect.left, rect.bottom, rect.width, rect.height]), 're'))
        page2content.operations.insert(1, ([], 'W'))
        page2content.operations.insert(2, ([], 'n'))
        if page2transformation is not None:
            page2content = page2transformation(page2content)
        page2content = PageObject._content_stream_rename(page2content, rename, self.pdf)
        page2content.isolate_graphics_state()
        if over:
            new_content_array.append(page2content)
        else:
            new_content_array.insert(0, page2content)
    if expand:
        self._expand_mediabox(page2, ctm)
    self.replace_contents(ContentStream(new_content_array, self.pdf))
    self[NameObject(PG.RESOURCES)] = new_resources
    self[NameObject(PG.ANNOTS)] = new_annots