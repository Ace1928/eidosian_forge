from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
class TypeAnnotationCode(proto.Enum):
    """``TypeAnnotationCode`` is used as a part of
    [Type][google.spanner.v1.Type] to disambiguate SQL types that should
    be used for a given Cloud Spanner value. Disambiguation is needed
    because the same Cloud Spanner type can be mapped to different SQL
    types depending on SQL dialect. TypeAnnotationCode doesn't affect
    the way value is serialized.

    Values:
        TYPE_ANNOTATION_CODE_UNSPECIFIED (0):
            Not specified.
        PG_NUMERIC (2):
            PostgreSQL compatible NUMERIC type. This annotation needs to
            be applied to [Type][google.spanner.v1.Type] instances
            having [NUMERIC][google.spanner.v1.TypeCode.NUMERIC] type
            code to specify that values of this type should be treated
            as PostgreSQL NUMERIC values. Currently this annotation is
            always needed for
            [NUMERIC][google.spanner.v1.TypeCode.NUMERIC] when a client
            interacts with PostgreSQL-enabled Spanner databases.
        PG_JSONB (3):
            PostgreSQL compatible JSONB type. This annotation needs to
            be applied to [Type][google.spanner.v1.Type] instances
            having [JSON][google.spanner.v1.TypeCode.JSON] type code to
            specify that values of this type should be treated as
            PostgreSQL JSONB values. Currently this annotation is always
            needed for [JSON][google.spanner.v1.TypeCode.JSON] when a
            client interacts with PostgreSQL-enabled Spanner databases.
    """
    TYPE_ANNOTATION_CODE_UNSPECIFIED = 0
    PG_NUMERIC = 2
    PG_JSONB = 3