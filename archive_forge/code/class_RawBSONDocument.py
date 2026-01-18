from __future__ import annotations
from typing import Any, ItemsView, Iterator, Mapping, MutableMapping, Optional
from bson import _get_object_size, _raw_to_dict
from bson.codec_options import _RAW_BSON_DOCUMENT_MARKER, CodecOptions
from bson.codec_options import DEFAULT_CODEC_OPTIONS as DEFAULT
from bson.son import SON
class RawBSONDocument(Mapping[str, Any]):
    """Representation for a MongoDB document that provides access to the raw
    BSON bytes that compose it.

    Only when a field is accessed or modified within the document does
    RawBSONDocument decode its bytes.
    """
    __slots__ = ('__raw', '__inflated_doc', '__codec_options')
    _type_marker = _RAW_BSON_DOCUMENT_MARKER
    __codec_options: CodecOptions[RawBSONDocument]

    def __init__(self, bson_bytes: bytes, codec_options: Optional[CodecOptions[RawBSONDocument]]=None) -> None:
        """Create a new :class:`RawBSONDocument`

        :class:`RawBSONDocument` is a representation of a BSON document that
        provides access to the underlying raw BSON bytes. Only when a field is
        accessed or modified within the document does RawBSONDocument decode
        its bytes.

        :class:`RawBSONDocument` implements the ``Mapping`` abstract base
        class from the standard library so it can be used like a read-only
        ``dict``::

            >>> from bson import encode
            >>> raw_doc = RawBSONDocument(encode({'_id': 'my_doc'}))
            >>> raw_doc.raw
            b'...'
            >>> raw_doc['_id']
            'my_doc'

        :Parameters:
          - `bson_bytes`: the BSON bytes that compose this document
          - `codec_options` (optional): An instance of
            :class:`~bson.codec_options.CodecOptions` whose ``document_class``
            must be :class:`RawBSONDocument`. The default is
            :attr:`DEFAULT_RAW_BSON_OPTIONS`.

        .. versionchanged:: 3.8
          :class:`RawBSONDocument` now validates that the ``bson_bytes``
          passed in represent a single bson document.

        .. versionchanged:: 3.5
          If a :class:`~bson.codec_options.CodecOptions` is passed in, its
          `document_class` must be :class:`RawBSONDocument`.
        """
        self.__raw = bson_bytes
        self.__inflated_doc: Optional[Mapping[str, Any]] = None
        if codec_options is None:
            codec_options = DEFAULT_RAW_BSON_OPTIONS
        elif not issubclass(codec_options.document_class, RawBSONDocument):
            raise TypeError(f'RawBSONDocument cannot use CodecOptions with document class {codec_options.document_class}')
        self.__codec_options = codec_options
        _get_object_size(bson_bytes, 0, len(bson_bytes))

    @property
    def raw(self) -> bytes:
        """The raw BSON bytes composing this document."""
        return self.__raw

    def items(self) -> ItemsView[str, Any]:
        """Lazily decode and iterate elements in this document."""
        return self.__inflated.items()

    @property
    def __inflated(self) -> Mapping[str, Any]:
        if self.__inflated_doc is None:
            self.__inflated_doc = self._inflate_bson(self.__raw, self.__codec_options)
        return self.__inflated_doc

    @staticmethod
    def _inflate_bson(bson_bytes: bytes, codec_options: CodecOptions[RawBSONDocument]) -> Mapping[str, Any]:
        return _inflate_bson(bson_bytes, codec_options)

    def __getitem__(self, item: str) -> Any:
        return self.__inflated[item]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__inflated)

    def __len__(self) -> int:
        return len(self.__inflated)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RawBSONDocument):
            return self.__raw == other.raw
        return NotImplemented

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.raw!r}, codec_options={self.__codec_options!r})'