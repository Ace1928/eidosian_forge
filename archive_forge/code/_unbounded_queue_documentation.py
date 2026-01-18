from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar
import attrs
from .. import _core
from .._deprecate import deprecated
from .._util import final
Return an :class:`UnboundedQueueStatistics` object containing debugging information.