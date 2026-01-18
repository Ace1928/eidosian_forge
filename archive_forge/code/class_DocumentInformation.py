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
class DocumentInformation(DictionaryObject):
    """
    A class representing the basic document metadata provided in a PDF File.
    This class is accessible through
    :py:class:`PdfReader.metadata<pypdf.PdfReader.metadata>`.

    All text properties of the document metadata have
    *two* properties, eg. author and author_raw. The non-raw property will
    always return a ``TextStringObject``, making it ideal for a case where
    the metadata is being displayed. The raw property can sometimes return
    a ``ByteStringObject``, if pypdf was unable to decode the string's
    text encoding; this requires additional safety in the caller and
    therefore is not as commonly accessed.
    """

    def __init__(self) -> None:
        DictionaryObject.__init__(self)

    def _get_text(self, key: str) -> Optional[str]:
        retval = self.get(key, None)
        if isinstance(retval, TextStringObject):
            return retval
        return None

    @property
    def title(self) -> Optional[str]:
        """
        Read-only property accessing the document's title.

        Returns a ``TextStringObject`` or ``None`` if the title is not
        specified.
        """
        return self._get_text(DI.TITLE) or self.get(DI.TITLE).get_object() if self.get(DI.TITLE) else None

    @property
    def title_raw(self) -> Optional[str]:
        """The "raw" version of title; can return a ``ByteStringObject``."""
        return self.get(DI.TITLE)

    @property
    def author(self) -> Optional[str]:
        """
        Read-only property accessing the document's author.

        Returns a ``TextStringObject`` or ``None`` if the author is not
        specified.
        """
        return self._get_text(DI.AUTHOR)

    @property
    def author_raw(self) -> Optional[str]:
        """The "raw" version of author; can return a ``ByteStringObject``."""
        return self.get(DI.AUTHOR)

    @property
    def subject(self) -> Optional[str]:
        """
        Read-only property accessing the document's subject.

        Returns a ``TextStringObject`` or ``None`` if the subject is not
        specified.
        """
        return self._get_text(DI.SUBJECT)

    @property
    def subject_raw(self) -> Optional[str]:
        """The "raw" version of subject; can return a ``ByteStringObject``."""
        return self.get(DI.SUBJECT)

    @property
    def creator(self) -> Optional[str]:
        """
        Read-only property accessing the document's creator.

        If the document was converted to PDF from another format, this is the
        name of the application (e.g. OpenOffice) that created the original
        document from which it was converted. Returns a ``TextStringObject`` or
        ``None`` if the creator is not specified.
        """
        return self._get_text(DI.CREATOR)

    @property
    def creator_raw(self) -> Optional[str]:
        """The "raw" version of creator; can return a ``ByteStringObject``."""
        return self.get(DI.CREATOR)

    @property
    def producer(self) -> Optional[str]:
        """
        Read-only property accessing the document's producer.

        If the document was converted to PDF from another format, this is the
        name of the application (for example, macOS Quartz) that converted it to
        PDF. Returns a ``TextStringObject`` or ``None`` if the producer is not
        specified.
        """
        return self._get_text(DI.PRODUCER)

    @property
    def producer_raw(self) -> Optional[str]:
        """The "raw" version of producer; can return a ``ByteStringObject``."""
        return self.get(DI.PRODUCER)

    @property
    def creation_date(self) -> Optional[datetime]:
        """Read-only property accessing the document's creation date."""
        return parse_iso8824_date(self._get_text(DI.CREATION_DATE))

    @property
    def creation_date_raw(self) -> Optional[str]:
        """
        The "raw" version of creation date; can return a ``ByteStringObject``.

        Typically in the format ``D:YYYYMMDDhhmmss[+Z-]hh'mm`` where the suffix
        is the offset from UTC.
        """
        return self.get(DI.CREATION_DATE)

    @property
    def modification_date(self) -> Optional[datetime]:
        """
        Read-only property accessing the document's modification date.

        The date and time the document was most recently modified.
        """
        return parse_iso8824_date(self._get_text(DI.MOD_DATE))

    @property
    def modification_date_raw(self) -> Optional[str]:
        """
        The "raw" version of modification date; can return a
        ``ByteStringObject``.

        Typically in the format ``D:YYYYMMDDhhmmss[+Z-]hh'mm`` where the suffix
        is the offset from UTC.
        """
        return self.get(DI.MOD_DATE)