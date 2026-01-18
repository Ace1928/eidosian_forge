import struct
import zlib
from abc import abstractmethod
from datetime import datetime
from typing import (
from ._encryption import Encryption
from ._page import PageObject, _VirtualList
from ._page_labels import index2label as page_index2page_label
from ._utils import (
from .constants import CatalogAttributes as CA
from .constants import CatalogDictionary as CD
from .constants import (
from .constants import Core as CO
from .constants import DocumentInformationAttributes as DI
from .constants import FieldDictionaryAttributes as FA
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .errors import (
from .generic import (
from .types import OutlineType, PagemodeType
from .xmp import XmpInformation
def _get_named_destinations(self, tree: Union[TreeObject, None]=None, retval: Optional[Any]=None) -> Dict[str, Any]:
    """
        Retrieve the named destinations present in the document.

        Args:
            tree:
            retval:

        Returns:
            A dictionary which maps names to
            :class:`Destinations<pypdf.generic.Destination>`.
        """
    if retval is None:
        retval = {}
        catalog = self.root_object
        if CA.DESTS in catalog:
            tree = cast(TreeObject, catalog[CA.DESTS])
        elif CA.NAMES in catalog:
            names = cast(DictionaryObject, catalog[CA.NAMES])
            if CA.DESTS in names:
                tree = cast(TreeObject, names[CA.DESTS])
    if tree is None:
        return retval
    if PA.KIDS in tree:
        for kid in cast(ArrayObject, tree[PA.KIDS]):
            self._get_named_destinations(kid.get_object(), retval)
    elif CA.NAMES in tree:
        names = cast(DictionaryObject, tree[CA.NAMES])
        i = 0
        while i < len(names):
            key = cast(str, names[i].get_object())
            i += 1
            if not isinstance(key, str):
                continue
            try:
                value = names[i].get_object()
            except IndexError:
                break
            i += 1
            if isinstance(value, DictionaryObject):
                if '/D' in value:
                    value = value['/D']
                else:
                    continue
            dest = self._build_destination(key, value)
            if dest is not None:
                retval[key] = dest
    else:
        for k__, v__ in tree.items():
            val = v__.get_object()
            if isinstance(val, DictionaryObject):
                if '/D' in val:
                    val = val['/D'].get_object()
                else:
                    continue
            dest = self._build_destination(k__, val)
            if dest is not None:
                retval[k__] = dest
    return retval