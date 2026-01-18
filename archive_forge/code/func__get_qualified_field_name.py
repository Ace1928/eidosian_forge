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
def _get_qualified_field_name(self, parent: DictionaryObject) -> str:
    if '/TM' in parent:
        return cast(str, parent['/TM'])
    elif '/Parent' in parent:
        return self._get_qualified_field_name(cast(DictionaryObject, parent['/Parent'])) + '.' + cast(str, parent.get('/T', ''))
    else:
        return cast(str, parent.get('/T', ''))