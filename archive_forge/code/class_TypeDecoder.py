from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
class TypeDecoder(abc.ABC):
    """Base class for defining type codec classes which describe how a
    BSON type can be transformed to a custom type.

    Codec classes must implement the ``bson_type`` attribute, and the
    ``transform_bson`` method to support decoding.

    See :ref:`custom-type-type-codec` documentation for an example.
    """

    @abc.abstractproperty
    def bson_type(self) -> Any:
        """The BSON type to be converted into our own type."""

    @abc.abstractmethod
    def transform_bson(self, value: Any) -> Any:
        """Convert the given BSON value into our own type."""