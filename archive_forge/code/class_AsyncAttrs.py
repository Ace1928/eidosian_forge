from __future__ import annotations
import asyncio
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import engine
from .base import ReversibleProxy
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import util
from ...orm import close_all_sessions as _sync_close_all_sessions
from ...orm import object_session
from ...orm import Session
from ...orm import SessionTransaction
from ...orm import state as _instance_state
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec
class AsyncAttrs:
    """Mixin class which provides an awaitable accessor for all attributes.

    E.g.::

        from __future__ import annotations

        from typing import List

        from sqlalchemy import ForeignKey
        from sqlalchemy import func
        from sqlalchemy.ext.asyncio import AsyncAttrs
        from sqlalchemy.orm import DeclarativeBase
        from sqlalchemy.orm import Mapped
        from sqlalchemy.orm import mapped_column
        from sqlalchemy.orm import relationship


        class Base(AsyncAttrs, DeclarativeBase):
            pass


        class A(Base):
            __tablename__ = "a"

            id: Mapped[int] = mapped_column(primary_key=True)
            data: Mapped[str]
            bs: Mapped[List[B]] = relationship()


        class B(Base):
            __tablename__ = "b"
            id: Mapped[int] = mapped_column(primary_key=True)
            a_id: Mapped[int] = mapped_column(ForeignKey("a.id"))
            data: Mapped[str]

    In the above example, the :class:`_asyncio.AsyncAttrs` mixin is applied to
    the declarative ``Base`` class where it takes effect for all subclasses.
    This mixin adds a single new attribute
    :attr:`_asyncio.AsyncAttrs.awaitable_attrs` to all classes, which will
    yield the value of any attribute as an awaitable. This allows attributes
    which may be subject to lazy loading or deferred / unexpiry loading to be
    accessed such that IO can still be emitted::

        a1 = (await async_session.scalars(select(A).where(A.id == 5))).one()

        # use the lazy loader on ``a1.bs`` via the ``.awaitable_attrs``
        # interface, so that it may be awaited
        for b1 in await a1.awaitable_attrs.bs:
            print(b1)

    The :attr:`_asyncio.AsyncAttrs.awaitable_attrs` performs a call against the
    attribute that is approximately equivalent to using the
    :meth:`_asyncio.AsyncSession.run_sync` method, e.g.::

        for b1 in await async_session.run_sync(lambda sess: a1.bs):
            print(b1)

    .. versionadded:: 2.0.13

    .. seealso::

        :ref:`asyncio_orm_avoid_lazyloads`

    """

    class _AsyncAttrGetitem:
        __slots__ = '_instance'

        def __init__(self, _instance: Any):
            self._instance = _instance

        def __getattr__(self, name: str) -> Awaitable[Any]:
            return greenlet_spawn(getattr, self._instance, name)

    @property
    def awaitable_attrs(self) -> AsyncAttrs._AsyncAttrGetitem:
        """provide a namespace of all attributes on this object wrapped
        as awaitables.

        e.g.::


            a1 = (await async_session.scalars(select(A).where(A.id == 5))).one()

            some_attribute = await a1.awaitable_attrs.some_deferred_attribute
            some_collection = await a1.awaitable_attrs.some_collection

        """
        return AsyncAttrs._AsyncAttrGetitem(self)