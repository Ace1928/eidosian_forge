from __future__ import annotations
from typing import Any
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql import bindparam
from . import attributes
from . import interfaces
from . import relationships
from . import strategies
from .base import NEVER_SET
from .base import object_mapper
from .base import PassiveFlag
from .base import RelationshipDirection
from .. import exc
from .. import inspect
from .. import log
from .. import util
from ..sql import delete
from ..sql import insert
from ..sql import select
from ..sql import update
from ..sql.dml import Delete
from ..sql.dml import Insert
from ..sql.dml import Update
from ..util.typing import Literal
class WriteOnlyCollection(AbstractCollectionWriter[_T]):
    """Write-only collection which can synchronize changes into the
    attribute event system.

    The :class:`.WriteOnlyCollection` is used in a mapping by
    using the ``"write_only"`` lazy loading strategy with
    :func:`_orm.relationship`.     For background on this configuration,
    see :ref:`write_only_relationship`.

    .. versionadded:: 2.0

    .. seealso::

        :ref:`write_only_relationship`

    """
    __slots__ = ('instance', 'attr', '_where_criteria', '_from_obj', '_order_by_clauses')

    def __iter__(self) -> NoReturn:
        raise TypeError("WriteOnly collections don't support iteration in-place; to query for collection items, use the select() method to produce a SQL statement and execute it with session.scalars().")

    def select(self) -> Select[Tuple[_T]]:
        """Produce a :class:`_sql.Select` construct that represents the
        rows within this instance-local :class:`_orm.WriteOnlyCollection`.

        """
        stmt = select(self.attr.target_mapper).where(*self._where_criteria)
        if self._from_obj:
            stmt = stmt.select_from(*self._from_obj)
        if self._order_by_clauses:
            stmt = stmt.order_by(*self._order_by_clauses)
        return stmt

    def insert(self) -> Insert:
        """For one-to-many collections, produce a :class:`_dml.Insert` which
        will insert new rows in terms of this this instance-local
        :class:`_orm.WriteOnlyCollection`.

        This construct is only supported for a :class:`_orm.Relationship`
        that does **not** include the :paramref:`_orm.relationship.secondary`
        parameter.  For relationships that refer to a many-to-many table,
        use ordinary bulk insert techniques to produce new objects, then
        use :meth:`_orm.AbstractCollectionWriter.add_all` to associate them
        with the collection.


        """
        state = inspect(self.instance)
        mapper = state.mapper
        prop = mapper._props[self.attr.key]
        if prop.direction is not RelationshipDirection.ONETOMANY:
            raise exc.InvalidRequestError('Write only bulk INSERT only supported for one-to-many collections; for many-to-many, use a separate bulk INSERT along with add_all().')
        dict_: Dict[str, Any] = {}
        for l, r in prop.synchronize_pairs:
            fn = prop._get_attr_w_warn_on_none(mapper, state, state.dict, l)
            dict_[r.key] = bindparam(None, callable_=fn)
        return insert(self.attr.target_mapper).values(**dict_)

    def update(self) -> Update:
        """Produce a :class:`_dml.Update` which will refer to rows in terms
        of this instance-local :class:`_orm.WriteOnlyCollection`.

        """
        return update(self.attr.target_mapper).where(*self._where_criteria)

    def delete(self) -> Delete:
        """Produce a :class:`_dml.Delete` which will refer to rows in terms
        of this instance-local :class:`_orm.WriteOnlyCollection`.

        """
        return delete(self.attr.target_mapper).where(*self._where_criteria)

    def add_all(self, iterator: Iterable[_T]) -> None:
        """Add an iterable of items to this :class:`_orm.WriteOnlyCollection`.

        The given items will be persisted to the database in terms of
        the parent instance's collection on the next flush.

        """
        self._add_all_impl(iterator)

    def add(self, item: _T) -> None:
        """Add an item to this :class:`_orm.WriteOnlyCollection`.

        The given item will be persisted to the database in terms of
        the parent instance's collection on the next flush.

        """
        self._add_all_impl([item])

    def remove(self, item: _T) -> None:
        """Remove an item from this :class:`_orm.WriteOnlyCollection`.

        The given item will be removed from the parent instance's collection on
        the next flush.

        """
        self._remove_impl(item)