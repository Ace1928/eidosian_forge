from __future__ import annotations
from typing import Any, ItemsView, Iterator, Mapping, MutableMapping, Optional
from bson import _get_object_size, _raw_to_dict
from bson.codec_options import _RAW_BSON_DOCUMENT_MARKER, CodecOptions
from bson.codec_options import DEFAULT_CODEC_OPTIONS as DEFAULT
from bson.son import SON
@property
def __inflated(self) -> Mapping[str, Any]:
    if self.__inflated_doc is None:
        self.__inflated_doc = self._inflate_bson(self.__raw, self.__codec_options)
    return self.__inflated_doc