from __future__ import annotations
import collections.abc as collections_abc
import datetime as dt
import decimal
import enum
import json
import pickle
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from uuid import UUID as _python_UUID
from . import coercions
from . import elements
from . import operators
from . import roles
from . import type_api
from .base import _NONE_NAME
from .base import NO_ARG
from .base import SchemaEventTarget
from .cache_key import HasCacheKey
from .elements import quoted_name
from .elements import Slice
from .elements import TypeCoerce as type_coerce  # noqa
from .type_api import Emulated
from .type_api import NativeForEmulated  # noqa
from .type_api import to_instance as to_instance
from .type_api import TypeDecorator as TypeDecorator
from .type_api import TypeEngine as TypeEngine
from .type_api import TypeEngineMixin
from .type_api import Variant  # noqa
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..engine import processors
from ..util import langhelpers
from ..util import OrderedDict
from ..util.typing import is_literal
from ..util.typing import Literal
from ..util.typing import typing_get_args
class Uuid(Emulated, TypeEngine[_UUID_RETURN]):
    """Represent a database agnostic UUID datatype.

    For backends that have no "native" UUID datatype, the value will
    make use of ``CHAR(32)`` and store the UUID as a 32-character alphanumeric
    hex string.

    For backends which are known to support ``UUID`` directly or a similar
    uuid-storing datatype such as SQL Server's ``UNIQUEIDENTIFIER``, a
    "native" mode enabled by default allows these types will be used on those
    backends.

    In its default mode of use, the :class:`_sqltypes.Uuid` datatype expects
    **Python uuid objects**, from the Python
    `uuid <https://docs.python.org/3/library/uuid.html>`_
    module::

        import uuid

        from sqlalchemy import Uuid
        from sqlalchemy import Table, Column, MetaData, String


        metadata_obj = MetaData()

        t = Table(
            "t",
            metadata_obj,
            Column('uuid_data', Uuid, primary_key=True),
            Column("other_data", String)
        )

        with engine.begin() as conn:
            conn.execute(
                t.insert(),
                {"uuid_data": uuid.uuid4(), "other_data", "some data"}
            )

    To have the :class:`_sqltypes.Uuid` datatype work with string-based
    Uuids (e.g. 32 character hexadecimal strings), pass the
    :paramref:`_sqltypes.Uuid.as_uuid` parameter with the value ``False``.

    .. versionadded:: 2.0

    .. seealso::

        :class:`_sqltypes.UUID` - represents exactly the ``UUID`` datatype
        without any backend-agnostic behaviors.

    """
    __visit_name__ = 'uuid'
    collation: Optional[str] = None

    @overload
    def __init__(self: Uuid[_python_UUID], as_uuid: Literal[True]=..., native_uuid: bool=...):
        ...

    @overload
    def __init__(self: Uuid[str], as_uuid: Literal[False]=..., native_uuid: bool=...):
        ...

    def __init__(self, as_uuid: bool=True, native_uuid: bool=True):
        """Construct a :class:`_sqltypes.Uuid` type.

        :param as_uuid=True: if True, values will be interpreted
         as Python uuid objects, converting to/from string via the
         DBAPI.

         .. versionchanged: 2.0 ``as_uuid`` now defaults to ``True``.

        :param native_uuid=True: if True, backends that support either the
         ``UUID`` datatype directly, or a UUID-storing value
         (such as SQL Server's ``UNIQUEIDENTIFIER`` will be used by those
         backends.   If False, a ``CHAR(32)`` datatype will be used for
         all backends regardless of native support.

        """
        self.as_uuid = as_uuid
        self.native_uuid = native_uuid

    @property
    def python_type(self):
        return _python_UUID if self.as_uuid else str

    @property
    def native(self):
        return self.native_uuid

    def coerce_compared_value(self, op, value):
        """See :meth:`.TypeEngine.coerce_compared_value` for a description."""
        if isinstance(value, str):
            return self
        else:
            return super().coerce_compared_value(op, value)

    def bind_processor(self, dialect):
        character_based_uuid = not dialect.supports_native_uuid or not self.native_uuid
        if character_based_uuid:
            if self.as_uuid:

                def process(value):
                    if value is not None:
                        value = value.hex
                    return value
                return process
            else:

                def process(value):
                    if value is not None:
                        value = value.replace('-', '')
                    return value
                return process
        else:
            return None

    def result_processor(self, dialect, coltype):
        character_based_uuid = not dialect.supports_native_uuid or not self.native_uuid
        if character_based_uuid:
            if self.as_uuid:

                def process(value):
                    if value is not None:
                        value = _python_UUID(value)
                    return value
                return process
            else:

                def process(value):
                    if value is not None:
                        value = str(_python_UUID(value))
                    return value
                return process
        elif not self.as_uuid:

            def process(value):
                if value is not None:
                    value = str(value)
                return value
            return process
        else:
            return None

    def literal_processor(self, dialect):
        character_based_uuid = not dialect.supports_native_uuid or not self.native_uuid
        if not self.as_uuid:

            def process(value):
                return f"'{value.replace('-', '').replace("'", "''")}'"
            return process
        elif character_based_uuid:

            def process(value):
                return f"'{value.hex}'"
            return process
        else:

            def process(value):
                return f"'{str(value).replace("'", "''")}'"
            return process