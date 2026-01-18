from __future__ import annotations
import collections
from enum import Enum
import itertools
from typing import AbstractSet
from typing import Any as TODO_Any
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import cache_key
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from . import visitors
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from ._typing import _TP
from ._typing import is_column_element
from ._typing import is_select_statement
from ._typing import is_subquery
from ._typing import is_table
from ._typing import is_text_clause
from .annotation import Annotated
from .annotation import SupportsCloneAnnotations
from .base import _clone
from .base import _cloned_difference
from .base import _cloned_intersection
from .base import _entity_namespace_key
from .base import _EntityNamespace
from .base import _expand_cloned
from .base import _from_objects
from .base import _generative
from .base import _never_select_column
from .base import _NoArg
from .base import _select_iterables
from .base import CacheableOptions
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DedupeColumnCollection
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .base import HasMemoized
from .base import Immutable
from .coercions import _document_text_coercion
from .elements import _anonymous_label
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ClauseList
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import DQLDMLClauseElement
from .elements import GroupedElement
from .elements import literal_column
from .elements import TableValuedColumn
from .elements import UnaryExpression
from .operators import OperatorType
from .sqltypes import NULLTYPE
from .visitors import _TraverseInternalsType
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
class HasCTE(roles.HasCTERole, SelectsRows):
    """Mixin that declares a class to include CTE support."""
    _has_ctes_traverse_internals: _TraverseInternalsType = [('_independent_ctes', InternalTraversal.dp_clauseelement_list), ('_independent_ctes_opts', InternalTraversal.dp_plain_obj)]
    _independent_ctes: Tuple[CTE, ...] = ()
    _independent_ctes_opts: Tuple[_CTEOpts, ...] = ()

    @_generative
    def add_cte(self, *ctes: CTE, nest_here: bool=False) -> Self:
        """Add one or more :class:`_sql.CTE` constructs to this statement.

        This method will associate the given :class:`_sql.CTE` constructs with
        the parent statement such that they will each be unconditionally
        rendered in the WITH clause of the final statement, even if not
        referenced elsewhere within the statement or any sub-selects.

        The optional :paramref:`.HasCTE.add_cte.nest_here` parameter when set
        to True will have the effect that each given :class:`_sql.CTE` will
        render in a WITH clause rendered directly along with this statement,
        rather than being moved to the top of the ultimate rendered statement,
        even if this statement is rendered as a subquery within a larger
        statement.

        This method has two general uses. One is to embed CTE statements that
        serve some purpose without being referenced explicitly, such as the use
        case of embedding a DML statement such as an INSERT or UPDATE as a CTE
        inline with a primary statement that may draw from its results
        indirectly.  The other is to provide control over the exact placement
        of a particular series of CTE constructs that should remain rendered
        directly in terms of a particular statement that may be nested in a
        larger statement.

        E.g.::

            from sqlalchemy import table, column, select
            t = table('t', column('c1'), column('c2'))

            ins = t.insert().values({"c1": "x", "c2": "y"}).cte()

            stmt = select(t).add_cte(ins)

        Would render::

            WITH anon_1 AS
            (INSERT INTO t (c1, c2) VALUES (:param_1, :param_2))
            SELECT t.c1, t.c2
            FROM t

        Above, the "anon_1" CTE is not referenced in the SELECT
        statement, however still accomplishes the task of running an INSERT
        statement.

        Similarly in a DML-related context, using the PostgreSQL
        :class:`_postgresql.Insert` construct to generate an "upsert"::

            from sqlalchemy import table, column
            from sqlalchemy.dialects.postgresql import insert

            t = table("t", column("c1"), column("c2"))

            delete_statement_cte = (
                t.delete().where(t.c.c1 < 1).cte("deletions")
            )

            insert_stmt = insert(t).values({"c1": 1, "c2": 2})
            update_statement = insert_stmt.on_conflict_do_update(
                index_elements=[t.c.c1],
                set_={
                    "c1": insert_stmt.excluded.c1,
                    "c2": insert_stmt.excluded.c2,
                },
            ).add_cte(delete_statement_cte)

            print(update_statement)

        The above statement renders as::

            WITH deletions AS
            (DELETE FROM t WHERE t.c1 < %(c1_1)s)
            INSERT INTO t (c1, c2) VALUES (%(c1)s, %(c2)s)
            ON CONFLICT (c1) DO UPDATE SET c1 = excluded.c1, c2 = excluded.c2

        .. versionadded:: 1.4.21

        :param \\*ctes: zero or more :class:`.CTE` constructs.

         .. versionchanged:: 2.0  Multiple CTE instances are accepted

        :param nest_here: if True, the given CTE or CTEs will be rendered
         as though they specified the :paramref:`.HasCTE.cte.nesting` flag
         to ``True`` when they were added to this :class:`.HasCTE`.
         Assuming the given CTEs are not referenced in an outer-enclosing
         statement as well, the CTEs given should render at the level of
         this statement when this flag is given.

         .. versionadded:: 2.0

         .. seealso::

            :paramref:`.HasCTE.cte.nesting`


        """
        opt = _CTEOpts(nest_here)
        for cte in ctes:
            cte = coercions.expect(roles.IsCTERole, cte)
            self._independent_ctes += (cte,)
            self._independent_ctes_opts += (opt,)
        return self

    def cte(self, name: Optional[str]=None, recursive: bool=False, nesting: bool=False) -> CTE:
        """Return a new :class:`_expression.CTE`,
        or Common Table Expression instance.

        Common table expressions are a SQL standard whereby SELECT
        statements can draw upon secondary statements specified along
        with the primary statement, using a clause called "WITH".
        Special semantics regarding UNION can also be employed to
        allow "recursive" queries, where a SELECT statement can draw
        upon the set of rows that have previously been selected.

        CTEs can also be applied to DML constructs UPDATE, INSERT
        and DELETE on some databases, both as a source of CTE rows
        when combined with RETURNING, as well as a consumer of
        CTE rows.

        SQLAlchemy detects :class:`_expression.CTE` objects, which are treated
        similarly to :class:`_expression.Alias` objects, as special elements
        to be delivered to the FROM clause of the statement as well
        as to a WITH clause at the top of the statement.

        For special prefixes such as PostgreSQL "MATERIALIZED" and
        "NOT MATERIALIZED", the :meth:`_expression.CTE.prefix_with`
        method may be
        used to establish these.

        .. versionchanged:: 1.3.13 Added support for prefixes.
           In particular - MATERIALIZED and NOT MATERIALIZED.

        :param name: name given to the common table expression.  Like
         :meth:`_expression.FromClause.alias`, the name can be left as
         ``None`` in which case an anonymous symbol will be used at query
         compile time.
        :param recursive: if ``True``, will render ``WITH RECURSIVE``.
         A recursive common table expression is intended to be used in
         conjunction with UNION ALL in order to derive rows
         from those already selected.
        :param nesting: if ``True``, will render the CTE locally to the
         statement in which it is referenced.   For more complex scenarios,
         the :meth:`.HasCTE.add_cte` method using the
         :paramref:`.HasCTE.add_cte.nest_here`
         parameter may also be used to more carefully
         control the exact placement of a particular CTE.

         .. versionadded:: 1.4.24

         .. seealso::

            :meth:`.HasCTE.add_cte`

        The following examples include two from PostgreSQL's documentation at
        https://www.postgresql.org/docs/current/static/queries-with.html,
        as well as additional examples.

        Example 1, non recursive::

            from sqlalchemy import (Table, Column, String, Integer,
                                    MetaData, select, func)

            metadata = MetaData()

            orders = Table('orders', metadata,
                Column('region', String),
                Column('amount', Integer),
                Column('product', String),
                Column('quantity', Integer)
            )

            regional_sales = select(
                                orders.c.region,
                                func.sum(orders.c.amount).label('total_sales')
                            ).group_by(orders.c.region).cte("regional_sales")


            top_regions = select(regional_sales.c.region).\\
                    where(
                        regional_sales.c.total_sales >
                        select(
                            func.sum(regional_sales.c.total_sales) / 10
                        )
                    ).cte("top_regions")

            statement = select(
                        orders.c.region,
                        orders.c.product,
                        func.sum(orders.c.quantity).label("product_units"),
                        func.sum(orders.c.amount).label("product_sales")
                ).where(orders.c.region.in_(
                    select(top_regions.c.region)
                )).group_by(orders.c.region, orders.c.product)

            result = conn.execute(statement).fetchall()

        Example 2, WITH RECURSIVE::

            from sqlalchemy import (Table, Column, String, Integer,
                                    MetaData, select, func)

            metadata = MetaData()

            parts = Table('parts', metadata,
                Column('part', String),
                Column('sub_part', String),
                Column('quantity', Integer),
            )

            included_parts = select(\\
                parts.c.sub_part, parts.c.part, parts.c.quantity\\
                ).\\
                where(parts.c.part=='our part').\\
                cte(recursive=True)


            incl_alias = included_parts.alias()
            parts_alias = parts.alias()
            included_parts = included_parts.union_all(
                select(
                    parts_alias.c.sub_part,
                    parts_alias.c.part,
                    parts_alias.c.quantity
                ).\\
                where(parts_alias.c.part==incl_alias.c.sub_part)
            )

            statement = select(
                        included_parts.c.sub_part,
                        func.sum(included_parts.c.quantity).
                          label('total_quantity')
                    ).\\
                    group_by(included_parts.c.sub_part)

            result = conn.execute(statement).fetchall()

        Example 3, an upsert using UPDATE and INSERT with CTEs::

            from datetime import date
            from sqlalchemy import (MetaData, Table, Column, Integer,
                                    Date, select, literal, and_, exists)

            metadata = MetaData()

            visitors = Table('visitors', metadata,
                Column('product_id', Integer, primary_key=True),
                Column('date', Date, primary_key=True),
                Column('count', Integer),
            )

            # add 5 visitors for the product_id == 1
            product_id = 1
            day = date.today()
            count = 5

            update_cte = (
                visitors.update()
                .where(and_(visitors.c.product_id == product_id,
                            visitors.c.date == day))
                .values(count=visitors.c.count + count)
                .returning(literal(1))
                .cte('update_cte')
            )

            upsert = visitors.insert().from_select(
                [visitors.c.product_id, visitors.c.date, visitors.c.count],
                select(literal(product_id), literal(day), literal(count))
                    .where(~exists(update_cte.select()))
            )

            connection.execute(upsert)

        Example 4, Nesting CTE (SQLAlchemy 1.4.24 and above)::

            value_a = select(
                literal("root").label("n")
            ).cte("value_a")

            # A nested CTE with the same name as the root one
            value_a_nested = select(
                literal("nesting").label("n")
            ).cte("value_a", nesting=True)

            # Nesting CTEs takes ascendency locally
            # over the CTEs at a higher level
            value_b = select(value_a_nested.c.n).cte("value_b")

            value_ab = select(value_a.c.n.label("a"), value_b.c.n.label("b"))

        The above query will render the second CTE nested inside the first,
        shown with inline parameters below as::

            WITH
                value_a AS
                    (SELECT 'root' AS n),
                value_b AS
                    (WITH value_a AS
                        (SELECT 'nesting' AS n)
                    SELECT value_a.n AS n FROM value_a)
            SELECT value_a.n AS a, value_b.n AS b
            FROM value_a, value_b

        The same CTE can be set up using the :meth:`.HasCTE.add_cte` method
        as follows (SQLAlchemy 2.0 and above)::

            value_a = select(
                literal("root").label("n")
            ).cte("value_a")

            # A nested CTE with the same name as the root one
            value_a_nested = select(
                literal("nesting").label("n")
            ).cte("value_a")

            # Nesting CTEs takes ascendency locally
            # over the CTEs at a higher level
            value_b = (
                select(value_a_nested.c.n).
                add_cte(value_a_nested, nest_here=True).
                cte("value_b")
            )

            value_ab = select(value_a.c.n.label("a"), value_b.c.n.label("b"))

        Example 5, Non-Linear CTE (SQLAlchemy 1.4.28 and above)::

            edge = Table(
                "edge",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("left", Integer),
                Column("right", Integer),
            )

            root_node = select(literal(1).label("node")).cte(
                "nodes", recursive=True
            )

            left_edge = select(edge.c.left).join(
                root_node, edge.c.right == root_node.c.node
            )
            right_edge = select(edge.c.right).join(
                root_node, edge.c.left == root_node.c.node
            )

            subgraph_cte = root_node.union(left_edge, right_edge)

            subgraph = select(subgraph_cte)

        The above query will render 2 UNIONs inside the recursive CTE::

            WITH RECURSIVE nodes(node) AS (
                    SELECT 1 AS node
                UNION
                    SELECT edge."left" AS "left"
                    FROM edge JOIN nodes ON edge."right" = nodes.node
                UNION
                    SELECT edge."right" AS "right"
                    FROM edge JOIN nodes ON edge."left" = nodes.node
            )
            SELECT nodes.node FROM nodes

        .. seealso::

            :meth:`_orm.Query.cte` - ORM version of
            :meth:`_expression.HasCTE.cte`.

        """
        return CTE._construct(self, name=name, recursive=recursive, nesting=nesting)