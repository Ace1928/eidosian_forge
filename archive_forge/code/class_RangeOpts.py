from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional
from bson import int64
from pymongo.common import validate_is_mapping
from pymongo.errors import ConfigurationError
from pymongo.uri_parser import _parse_kms_tls_options
class RangeOpts:
    """Options to configure encrypted queries using the rangePreview algorithm."""

    def __init__(self, sparsity: int, min: Optional[Any]=None, max: Optional[Any]=None, precision: Optional[int]=None) -> None:
        """Options to configure encrypted queries using the rangePreview algorithm.

        .. note:: This feature is experimental only, and not intended for public use.

        :Parameters:
          - `sparsity`: An integer.
          - `min`: A BSON scalar value corresponding to the type being queried.
          - `max`: A BSON scalar value corresponding to the type being queried.
          - `precision`: An integer, may only be set for double or decimal128 types.

        .. versionadded:: 4.4
        """
        self.min = min
        self.max = max
        self.sparsity = sparsity
        self.precision = precision

    @property
    def document(self) -> dict[str, Any]:
        doc = {}
        for k, v in [('sparsity', int64.Int64(self.sparsity)), ('precision', self.precision), ('min', self.min), ('max', self.max)]:
            if v is not None:
                doc[k] = v
        return doc