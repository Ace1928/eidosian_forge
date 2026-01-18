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
class IdentityOptions:
    """Defines options for a named database sequence or an identity column.

    .. versionadded:: 1.3.18

    .. seealso::

        :class:`.Sequence`

    """

    def __init__(self, start: Optional[int]=None, increment: Optional[int]=None, minvalue: Optional[int]=None, maxvalue: Optional[int]=None, nominvalue: Optional[bool]=None, nomaxvalue: Optional[bool]=None, cycle: Optional[bool]=None, cache: Optional[int]=None, order: Optional[bool]=None) -> None:
        """Construct a :class:`.IdentityOptions` object.

        See the :class:`.Sequence` documentation for a complete description
        of the parameters.

        :param start: the starting index of the sequence.
        :param increment: the increment value of the sequence.
        :param minvalue: the minimum value of the sequence.
        :param maxvalue: the maximum value of the sequence.
        :param nominvalue: no minimum value of the sequence.
        :param nomaxvalue: no maximum value of the sequence.
        :param cycle: allows the sequence to wrap around when the maxvalue
         or minvalue has been reached.
        :param cache: optional integer value; number of future values in the
         sequence which are calculated in advance.
        :param order: optional boolean value; if ``True``, renders the
         ORDER keyword.

        """
        self.start = start
        self.increment = increment
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.nominvalue = nominvalue
        self.nomaxvalue = nomaxvalue
        self.cycle = cycle
        self.cache = cache
        self.order = order

    @property
    def _increment_is_negative(self) -> bool:
        return self.increment is not None and self.increment < 0