from __future__ import annotations
from typing import TYPE_CHECKING
from .dml import Delete
from .dml import Insert
from .dml import Update
Construct :class:`_expression.Delete` object.

    E.g.::

        from sqlalchemy import delete

        stmt = (
            delete(user_table).
            where(user_table.c.id == 5)
        )

    Similar functionality is available via the
    :meth:`_expression.TableClause.delete` method on
    :class:`_schema.Table`.

    :param table: The table to delete rows from.

    .. seealso::

        :ref:`tutorial_core_update_delete` - in the :ref:`unified_tutorial`


    