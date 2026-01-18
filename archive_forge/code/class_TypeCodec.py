from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
class TypeCodec(TypeEncoder, TypeDecoder):
    """Base class for defining type codec classes which describe how a
    custom type can be transformed to/from one of the types :mod:`bson`
    can already encode/decode.

    Codec classes must implement the ``python_type`` attribute, and the
    ``transform_python`` method to support encoding, as well as the
    ``bson_type`` attribute, and the ``transform_bson`` method to support
    decoding.

    See :ref:`custom-type-type-codec` documentation for an example.
    """