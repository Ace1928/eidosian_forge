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
def _merge_resources(self, res1: DictionaryObject, res2: DictionaryObject, resource: Any, new_res1: bool=True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        assert isinstance(self.indirect_reference, IndirectObject)
        pdf = self.indirect_reference.pdf
        is_pdf_writer = hasattr(pdf, '_add_object')
    except (AssertionError, AttributeError):
        pdf = None
        is_pdf_writer = False

    def compute_unique_key(base_key: str) -> Tuple[str, bool]:
        """
            Find a key that either doesn't already exist or has the same value
            (indicated by the bool)

            Args:
                base_key: An index is added to this to get the computed key

            Returns:
                A tuple (computed key, bool) where the boolean indicates
                if there is a resource of the given computed_key with the same
                value.
            """
        value = page2res.raw_get(base_key)
        computed_key = base_key
        idx = 0
        while computed_key in new_res:
            if new_res.raw_get(computed_key) == value:
                return (computed_key, True)
            computed_key = f'{base_key}-{idx}'
            idx += 1
        return (computed_key, False)
    if new_res1:
        new_res = DictionaryObject()
        new_res.update(res1.get(resource, DictionaryObject()).get_object())
    else:
        new_res = cast(DictionaryObject, res1[resource])
    page2res = cast(DictionaryObject, res2.get(resource, DictionaryObject()).get_object())
    rename_res = {}
    for key in page2res:
        unique_key, same_value = compute_unique_key(key)
        newname = NameObject(unique_key)
        if key != unique_key:
            rename_res[key] = newname
        if not same_value:
            if is_pdf_writer:
                new_res[newname] = page2res.raw_get(key).clone(pdf)
                try:
                    new_res[newname] = new_res[newname].indirect_reference
                except AttributeError:
                    pass
            else:
                new_res[newname] = page2res.raw_get(key)
        lst = sorted(new_res.items())
        new_res.clear()
        for el in lst:
            new_res[el[0]] = el[1]
    return (new_res, rename_res)