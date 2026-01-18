from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple, Type, Union
from uuid import UUID
class UuidRepresentation:
    UNSPECIFIED = 0
    'An unspecified UUID representation.\n\n    When configured, :class:`uuid.UUID` instances will **not** be\n    automatically encoded to or decoded from :class:`~bson.binary.Binary`.\n    When encoding a :class:`uuid.UUID` instance, an error will be raised.\n    To encode a :class:`uuid.UUID` instance with this configuration, it must\n    be wrapped in the :class:`~bson.binary.Binary` class by the application\n    code. When decoding a BSON binary field with a UUID subtype, a\n    :class:`~bson.binary.Binary` instance will be returned instead of a\n    :class:`uuid.UUID` instance.\n\n    See :ref:`unspecified-representation-details` for details.\n\n    .. versionadded:: 3.11\n    '
    STANDARD = UUID_SUBTYPE
    'The standard UUID representation.\n\n    :class:`uuid.UUID` instances will automatically be encoded to\n    and decoded from BSON binary, using RFC-4122 byte order with\n    binary subtype :data:`UUID_SUBTYPE`.\n\n    See :ref:`standard-representation-details` for details.\n\n    .. versionadded:: 3.11\n    '
    PYTHON_LEGACY = OLD_UUID_SUBTYPE
    'The Python legacy UUID representation.\n\n    :class:`uuid.UUID` instances will automatically be encoded to\n    and decoded from BSON binary, using RFC-4122 byte order with\n    binary subtype :data:`OLD_UUID_SUBTYPE`.\n\n    See :ref:`python-legacy-representation-details` for details.\n\n    .. versionadded:: 3.11\n    '
    JAVA_LEGACY = 5
    "The Java legacy UUID representation.\n\n    :class:`uuid.UUID` instances will automatically be encoded to\n    and decoded from BSON binary subtype :data:`OLD_UUID_SUBTYPE`,\n    using the Java driver's legacy byte order.\n\n    See :ref:`java-legacy-representation-details` for details.\n\n    .. versionadded:: 3.11\n    "
    CSHARP_LEGACY = 6
    "The C#/.net legacy UUID representation.\n\n    :class:`uuid.UUID` instances will automatically be encoded to\n    and decoded from BSON binary subtype :data:`OLD_UUID_SUBTYPE`,\n    using the C# driver's legacy byte order.\n\n    See :ref:`csharp-legacy-representation-details` for details.\n\n    .. versionadded:: 3.11\n    "