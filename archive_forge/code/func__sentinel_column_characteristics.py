from __future__ import annotations
from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
@util.ro_memoized_property
def _sentinel_column_characteristics(self) -> _SentinelColumnCharacterization:
    """determine a candidate column (or columns, in case of a client
        generated composite primary key) which can be used as an
        "insert sentinel" for an INSERT statement.

        The returned structure, :class:`_SentinelColumnCharacterization`,
        includes all the details needed by :class:`.Dialect` and
        :class:`.SQLCompiler` to determine if these column(s) can be used
        as an INSERT..RETURNING sentinel for a particular database
        dialect.

        .. versionadded:: 2.0.10

        """
    sentinel_is_explicit = False
    sentinel_is_autoinc = False
    the_sentinel: Optional[_typing_Sequence[Column[Any]]] = None
    explicit_sentinel_col = self._sentinel_column
    if explicit_sentinel_col is not None:
        the_sentinel = (explicit_sentinel_col,)
        sentinel_is_explicit = True
    autoinc_col = self._autoincrement_column
    if sentinel_is_explicit and explicit_sentinel_col is autoinc_col:
        assert autoinc_col is not None
        sentinel_is_autoinc = True
    elif explicit_sentinel_col is None and autoinc_col is not None:
        the_sentinel = (autoinc_col,)
        sentinel_is_autoinc = True
    default_characterization = _SentinelDefaultCharacterization.UNKNOWN
    if the_sentinel:
        the_sentinel_zero = the_sentinel[0]
        if the_sentinel_zero.identity:
            if the_sentinel_zero.identity._increment_is_negative:
                if sentinel_is_explicit:
                    raise exc.InvalidRequestError("Can't use IDENTITY default with negative increment as an explicit sentinel column")
                else:
                    if sentinel_is_autoinc:
                        autoinc_col = None
                        sentinel_is_autoinc = False
                    the_sentinel = None
            else:
                default_characterization = _SentinelDefaultCharacterization.IDENTITY
        elif the_sentinel_zero.default is None and the_sentinel_zero.server_default is None:
            if the_sentinel_zero.nullable:
                raise exc.InvalidRequestError(f'Column {the_sentinel_zero} has been marked as a sentinel column with no default generation function; it at least needs to be marked nullable=False assuming user-populated sentinel values will be used.')
            default_characterization = _SentinelDefaultCharacterization.NONE
        elif the_sentinel_zero.default is not None:
            if the_sentinel_zero.default.is_sentinel:
                default_characterization = _SentinelDefaultCharacterization.SENTINEL_DEFAULT
            elif default_is_sequence(the_sentinel_zero.default):
                if the_sentinel_zero.default._increment_is_negative:
                    if sentinel_is_explicit:
                        raise exc.InvalidRequestError("Can't use SEQUENCE default with negative increment as an explicit sentinel column")
                    else:
                        if sentinel_is_autoinc:
                            autoinc_col = None
                            sentinel_is_autoinc = False
                        the_sentinel = None
                default_characterization = _SentinelDefaultCharacterization.SEQUENCE
            elif the_sentinel_zero.default.is_callable:
                default_characterization = _SentinelDefaultCharacterization.CLIENTSIDE
        elif the_sentinel_zero.server_default is not None:
            if sentinel_is_explicit:
                raise exc.InvalidRequestError(f"Column {the_sentinel[0]} can't be a sentinel column because it uses an explicit server side default that's not the Identity() default.")
            default_characterization = _SentinelDefaultCharacterization.SERVERSIDE
    if the_sentinel is None and self.primary_key:
        assert autoinc_col is None
        for _pkc in self.primary_key:
            if _pkc.server_default is not None or (_pkc.default and (not _pkc.default.is_callable)):
                break
        else:
            the_sentinel = tuple(self.primary_key)
            default_characterization = _SentinelDefaultCharacterization.CLIENTSIDE
    return _SentinelColumnCharacterization(the_sentinel, sentinel_is_explicit, sentinel_is_autoinc, default_characterization)