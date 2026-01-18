from __future__ import annotations
from typing import Any, ItemsView, Iterator, Mapping, MutableMapping, Optional
from bson import _get_object_size, _raw_to_dict
from bson.codec_options import _RAW_BSON_DOCUMENT_MARKER, CodecOptions
from bson.codec_options import DEFAULT_CODEC_OPTIONS as DEFAULT
from bson.son import SON
class _RawArrayBSONDocument(RawBSONDocument):
    """A RawBSONDocument that only expands sub-documents and arrays when accessed."""

    @staticmethod
    def _inflate_bson(bson_bytes: bytes, codec_options: CodecOptions[RawBSONDocument]) -> Mapping[str, Any]:
        return _inflate_bson(bson_bytes, codec_options, raw_array=True)