from __future__ import annotations
import collections
from collections import abc
import dataclasses
import inspect as _py_inspect
import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import strategy_options
from ._typing import insp_is_aliased_class
from ._typing import is_has_collection_adapter
from .base import _DeclarativeMapped
from .base import _is_mapped_class
from .base import class_mapper
from .base import DynamicMapped
from .base import LoaderCallableStatus
from .base import PassiveFlag
from .base import state_str
from .base import WriteOnlyMapped
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .interfaces import PropComparator
from .interfaces import RelationshipDirection
from .interfaces import StrategizedProperty
from .util import _orm_annotate
from .util import _orm_deannotate
from .util import CascadeOptions
from .. import exc as sa_exc
from .. import Exists
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..inspection import inspect
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql._typing import _ColumnExpressionArgument
from ..sql._typing import _HasClauseElement
from ..sql.annotation import _safe_annotate
from ..sql.elements import ColumnClause
from ..sql.elements import ColumnElement
from ..sql.util import _deep_annotate
from ..sql.util import _deep_deannotate
from ..sql.util import _shallow_annotate
from ..sql.util import adapt_criterion_to_null
from ..sql.util import ClauseAdapter
from ..sql.util import join_condition
from ..sql.util import selectables_overlap
from ..sql.util import visit_binary_product
from ..util.typing import de_optionalize_union_types
from ..util.typing import Literal
from ..util.typing import resolve_name_to_real_class_name
class JoinCondition:
    primaryjoin_initial: Optional[ColumnElement[bool]]
    primaryjoin: ColumnElement[bool]
    secondaryjoin: Optional[ColumnElement[bool]]
    secondary: Optional[FromClause]
    prop: RelationshipProperty[Any]
    synchronize_pairs: _ColumnPairs
    secondary_synchronize_pairs: _ColumnPairs
    direction: RelationshipDirection
    parent_persist_selectable: FromClause
    child_persist_selectable: FromClause
    parent_local_selectable: FromClause
    child_local_selectable: FromClause
    _local_remote_pairs: Optional[_ColumnPairs]

    def __init__(self, parent_persist_selectable: FromClause, child_persist_selectable: FromClause, parent_local_selectable: FromClause, child_local_selectable: FromClause, *, primaryjoin: Optional[ColumnElement[bool]]=None, secondary: Optional[FromClause]=None, secondaryjoin: Optional[ColumnElement[bool]]=None, parent_equivalents: Optional[_EquivalentColumnMap]=None, child_equivalents: Optional[_EquivalentColumnMap]=None, consider_as_foreign_keys: Any=None, local_remote_pairs: Optional[_ColumnPairs]=None, remote_side: Any=None, self_referential: Any=False, prop: RelationshipProperty[Any], support_sync: bool=True, can_be_synced_fn: Callable[..., bool]=lambda *c: True):
        self.parent_persist_selectable = parent_persist_selectable
        self.parent_local_selectable = parent_local_selectable
        self.child_persist_selectable = child_persist_selectable
        self.child_local_selectable = child_local_selectable
        self.parent_equivalents = parent_equivalents
        self.child_equivalents = child_equivalents
        self.primaryjoin_initial = primaryjoin
        self.secondaryjoin = secondaryjoin
        self.secondary = secondary
        self.consider_as_foreign_keys = consider_as_foreign_keys
        self._local_remote_pairs = local_remote_pairs
        self._remote_side = remote_side
        self.prop = prop
        self.self_referential = self_referential
        self.support_sync = support_sync
        self.can_be_synced_fn = can_be_synced_fn
        self._determine_joins()
        assert self.primaryjoin is not None
        self._sanitize_joins()
        self._annotate_fks()
        self._annotate_remote()
        self._annotate_local()
        self._annotate_parentmapper()
        self._setup_pairs()
        self._check_foreign_cols(self.primaryjoin, True)
        if self.secondaryjoin is not None:
            self._check_foreign_cols(self.secondaryjoin, False)
        self._determine_direction()
        self._check_remote_side()
        self._log_joins()

    def _log_joins(self) -> None:
        log = self.prop.logger
        log.info('%s setup primary join %s', self.prop, self.primaryjoin)
        log.info('%s setup secondary join %s', self.prop, self.secondaryjoin)
        log.info('%s synchronize pairs [%s]', self.prop, ','.join(('(%s => %s)' % (l, r) for l, r in self.synchronize_pairs)))
        log.info('%s secondary synchronize pairs [%s]', self.prop, ','.join(('(%s => %s)' % (l, r) for l, r in self.secondary_synchronize_pairs or [])))
        log.info('%s local/remote pairs [%s]', self.prop, ','.join(('(%s / %s)' % (l, r) for l, r in self.local_remote_pairs)))
        log.info('%s remote columns [%s]', self.prop, ','.join(('%s' % col for col in self.remote_columns)))
        log.info('%s local columns [%s]', self.prop, ','.join(('%s' % col for col in self.local_columns)))
        log.info('%s relationship direction %s', self.prop, self.direction)

    def _sanitize_joins(self) -> None:
        """remove the parententity annotation from our join conditions which
        can leak in here based on some declarative patterns and maybe others.

        "parentmapper" is relied upon both by the ORM evaluator as well as
        the use case in _join_fixture_inh_selfref_w_entity
        that relies upon it being present, see :ticket:`3364`.

        """
        self.primaryjoin = _deep_deannotate(self.primaryjoin, values=('parententity', 'proxy_key'))
        if self.secondaryjoin is not None:
            self.secondaryjoin = _deep_deannotate(self.secondaryjoin, values=('parententity', 'proxy_key'))

    def _determine_joins(self) -> None:
        """Determine the 'primaryjoin' and 'secondaryjoin' attributes,
        if not passed to the constructor already.

        This is based on analysis of the foreign key relationships
        between the parent and target mapped selectables.

        """
        if self.secondaryjoin is not None and self.secondary is None:
            raise sa_exc.ArgumentError('Property %s specified with secondary join condition but no secondary argument' % self.prop)
        try:
            consider_as_foreign_keys = self.consider_as_foreign_keys or None
            if self.secondary is not None:
                if self.secondaryjoin is None:
                    self.secondaryjoin = join_condition(self.child_persist_selectable, self.secondary, a_subset=self.child_local_selectable, consider_as_foreign_keys=consider_as_foreign_keys)
                if self.primaryjoin_initial is None:
                    self.primaryjoin = join_condition(self.parent_persist_selectable, self.secondary, a_subset=self.parent_local_selectable, consider_as_foreign_keys=consider_as_foreign_keys)
                else:
                    self.primaryjoin = self.primaryjoin_initial
            elif self.primaryjoin_initial is None:
                self.primaryjoin = join_condition(self.parent_persist_selectable, self.child_persist_selectable, a_subset=self.parent_local_selectable, consider_as_foreign_keys=consider_as_foreign_keys)
            else:
                self.primaryjoin = self.primaryjoin_initial
        except sa_exc.NoForeignKeysError as nfe:
            if self.secondary is not None:
                raise sa_exc.NoForeignKeysError("Could not determine join condition between parent/child tables on relationship %s - there are no foreign keys linking these tables via secondary table '%s'.  Ensure that referencing columns are associated with a ForeignKey or ForeignKeyConstraint, or specify 'primaryjoin' and 'secondaryjoin' expressions." % (self.prop, self.secondary)) from nfe
            else:
                raise sa_exc.NoForeignKeysError("Could not determine join condition between parent/child tables on relationship %s - there are no foreign keys linking these tables.  Ensure that referencing columns are associated with a ForeignKey or ForeignKeyConstraint, or specify a 'primaryjoin' expression." % self.prop) from nfe
        except sa_exc.AmbiguousForeignKeysError as afe:
            if self.secondary is not None:
                raise sa_exc.AmbiguousForeignKeysError("Could not determine join condition between parent/child tables on relationship %s - there are multiple foreign key paths linking the tables via secondary table '%s'.  Specify the 'foreign_keys' argument, providing a list of those columns which should be counted as containing a foreign key reference from the secondary table to each of the parent and child tables." % (self.prop, self.secondary)) from afe
            else:
                raise sa_exc.AmbiguousForeignKeysError("Could not determine join condition between parent/child tables on relationship %s - there are multiple foreign key paths linking the tables.  Specify the 'foreign_keys' argument, providing a list of those columns which should be counted as containing a foreign key reference to the parent table." % self.prop) from afe

    @property
    def primaryjoin_minus_local(self) -> ColumnElement[bool]:
        return _deep_deannotate(self.primaryjoin, values=('local', 'remote'))

    @property
    def secondaryjoin_minus_local(self) -> ColumnElement[bool]:
        assert self.secondaryjoin is not None
        return _deep_deannotate(self.secondaryjoin, values=('local', 'remote'))

    @util.memoized_property
    def primaryjoin_reverse_remote(self) -> ColumnElement[bool]:
        """Return the primaryjoin condition suitable for the
        "reverse" direction.

        If the primaryjoin was delivered here with pre-existing
        "remote" annotations, the local/remote annotations
        are reversed.  Otherwise, the local/remote annotations
        are removed.

        """
        if self._has_remote_annotations:

            def replace(element: _CE, **kw: Any) -> Optional[_CE]:
                if 'remote' in element._annotations:
                    v = dict(element._annotations)
                    del v['remote']
                    v['local'] = True
                    return element._with_annotations(v)
                elif 'local' in element._annotations:
                    v = dict(element._annotations)
                    del v['local']
                    v['remote'] = True
                    return element._with_annotations(v)
                return None
            return visitors.replacement_traverse(self.primaryjoin, {}, replace)
        elif self._has_foreign_annotations:
            return _deep_deannotate(self.primaryjoin, values=('local', 'remote'))
        else:
            return _deep_deannotate(self.primaryjoin)

    def _has_annotation(self, clause: ClauseElement, annotation: str) -> bool:
        for col in visitors.iterate(clause, {}):
            if annotation in col._annotations:
                return True
        else:
            return False

    @util.memoized_property
    def _has_foreign_annotations(self) -> bool:
        return self._has_annotation(self.primaryjoin, 'foreign')

    @util.memoized_property
    def _has_remote_annotations(self) -> bool:
        return self._has_annotation(self.primaryjoin, 'remote')

    def _annotate_fks(self) -> None:
        """Annotate the primaryjoin and secondaryjoin
        structures with 'foreign' annotations marking columns
        considered as foreign.

        """
        if self._has_foreign_annotations:
            return
        if self.consider_as_foreign_keys:
            self._annotate_from_fk_list()
        else:
            self._annotate_present_fks()

    def _annotate_from_fk_list(self) -> None:

        def check_fk(element: _CE, **kw: Any) -> Optional[_CE]:
            if element in self.consider_as_foreign_keys:
                return element._annotate({'foreign': True})
            return None
        self.primaryjoin = visitors.replacement_traverse(self.primaryjoin, {}, check_fk)
        if self.secondaryjoin is not None:
            self.secondaryjoin = visitors.replacement_traverse(self.secondaryjoin, {}, check_fk)

    def _annotate_present_fks(self) -> None:
        if self.secondary is not None:
            secondarycols = util.column_set(self.secondary.c)
        else:
            secondarycols = set()

        def is_foreign(a: ColumnElement[Any], b: ColumnElement[Any]) -> Optional[ColumnElement[Any]]:
            if isinstance(a, schema.Column) and isinstance(b, schema.Column):
                if a.references(b):
                    return a
                elif b.references(a):
                    return b
            if secondarycols:
                if a in secondarycols and b not in secondarycols:
                    return a
                elif b in secondarycols and a not in secondarycols:
                    return b
            return None

        def visit_binary(binary: BinaryExpression[Any]) -> None:
            if not isinstance(binary.left, sql.ColumnElement) or not isinstance(binary.right, sql.ColumnElement):
                return
            if 'foreign' not in binary.left._annotations and 'foreign' not in binary.right._annotations:
                col = is_foreign(binary.left, binary.right)
                if col is not None:
                    if col.compare(binary.left):
                        binary.left = binary.left._annotate({'foreign': True})
                    elif col.compare(binary.right):
                        binary.right = binary.right._annotate({'foreign': True})
        self.primaryjoin = visitors.cloned_traverse(self.primaryjoin, {}, {'binary': visit_binary})
        if self.secondaryjoin is not None:
            self.secondaryjoin = visitors.cloned_traverse(self.secondaryjoin, {}, {'binary': visit_binary})

    def _refers_to_parent_table(self) -> bool:
        """Return True if the join condition contains column
        comparisons where both columns are in both tables.

        """
        pt = self.parent_persist_selectable
        mt = self.child_persist_selectable
        result = False

        def visit_binary(binary: BinaryExpression[Any]) -> None:
            nonlocal result
            c, f = (binary.left, binary.right)
            if isinstance(c, expression.ColumnClause) and isinstance(f, expression.ColumnClause) and pt.is_derived_from(c.table) and pt.is_derived_from(f.table) and mt.is_derived_from(c.table) and mt.is_derived_from(f.table):
                result = True
        visitors.traverse(self.primaryjoin, {}, {'binary': visit_binary})
        return result

    def _tables_overlap(self) -> bool:
        """Return True if parent/child tables have some overlap."""
        return selectables_overlap(self.parent_persist_selectable, self.child_persist_selectable)

    def _annotate_remote(self) -> None:
        """Annotate the primaryjoin and secondaryjoin
        structures with 'remote' annotations marking columns
        considered as part of the 'remote' side.

        """
        if self._has_remote_annotations:
            return
        if self.secondary is not None:
            self._annotate_remote_secondary()
        elif self._local_remote_pairs or self._remote_side:
            self._annotate_remote_from_args()
        elif self._refers_to_parent_table():
            self._annotate_selfref(lambda col: 'foreign' in col._annotations, False)
        elif self._tables_overlap():
            self._annotate_remote_with_overlap()
        else:
            self._annotate_remote_distinct_selectables()

    def _annotate_remote_secondary(self) -> None:
        """annotate 'remote' in primaryjoin, secondaryjoin
        when 'secondary' is present.

        """
        assert self.secondary is not None
        fixed_secondary = self.secondary

        def repl(element: _CE, **kw: Any) -> Optional[_CE]:
            if fixed_secondary.c.contains_column(element):
                return element._annotate({'remote': True})
            return None
        self.primaryjoin = visitors.replacement_traverse(self.primaryjoin, {}, repl)
        assert self.secondaryjoin is not None
        self.secondaryjoin = visitors.replacement_traverse(self.secondaryjoin, {}, repl)

    def _annotate_selfref(self, fn: Callable[[ColumnElement[Any]], bool], remote_side_given: bool) -> None:
        """annotate 'remote' in primaryjoin, secondaryjoin
        when the relationship is detected as self-referential.

        """

        def visit_binary(binary: BinaryExpression[Any]) -> None:
            equated = binary.left.compare(binary.right)
            if isinstance(binary.left, expression.ColumnClause) and isinstance(binary.right, expression.ColumnClause):
                if fn(binary.left):
                    binary.left = binary.left._annotate({'remote': True})
                if fn(binary.right) and (not equated):
                    binary.right = binary.right._annotate({'remote': True})
            elif not remote_side_given:
                self._warn_non_column_elements()
        self.primaryjoin = visitors.cloned_traverse(self.primaryjoin, {}, {'binary': visit_binary})

    def _annotate_remote_from_args(self) -> None:
        """annotate 'remote' in primaryjoin, secondaryjoin
        when the 'remote_side' or '_local_remote_pairs'
        arguments are used.

        """
        if self._local_remote_pairs:
            if self._remote_side:
                raise sa_exc.ArgumentError('remote_side argument is redundant against more detailed _local_remote_side argument.')
            remote_side = [r for l, r in self._local_remote_pairs]
        else:
            remote_side = self._remote_side
        if self._refers_to_parent_table():
            self._annotate_selfref(lambda col: col in remote_side, True)
        else:

            def repl(element: _CE, **kw: Any) -> Optional[_CE]:
                if element in set(remote_side):
                    return element._annotate({'remote': True})
                return None
            self.primaryjoin = visitors.replacement_traverse(self.primaryjoin, {}, repl)

    def _annotate_remote_with_overlap(self) -> None:
        """annotate 'remote' in primaryjoin, secondaryjoin
        when the parent/child tables have some set of
        tables in common, though is not a fully self-referential
        relationship.

        """

        def visit_binary(binary: BinaryExpression[Any]) -> None:
            binary.left, binary.right = proc_left_right(binary.left, binary.right)
            binary.right, binary.left = proc_left_right(binary.right, binary.left)
        check_entities = self.prop is not None and self.prop.mapper is not self.prop.parent

        def proc_left_right(left: ColumnElement[Any], right: ColumnElement[Any]) -> Tuple[ColumnElement[Any], ColumnElement[Any]]:
            if isinstance(left, expression.ColumnClause) and isinstance(right, expression.ColumnClause):
                if self.child_persist_selectable.c.contains_column(right) and self.parent_persist_selectable.c.contains_column(left):
                    right = right._annotate({'remote': True})
            elif check_entities and right._annotations.get('parentmapper') is self.prop.mapper:
                right = right._annotate({'remote': True})
            elif check_entities and left._annotations.get('parentmapper') is self.prop.mapper:
                left = left._annotate({'remote': True})
            else:
                self._warn_non_column_elements()
            return (left, right)
        self.primaryjoin = visitors.cloned_traverse(self.primaryjoin, {}, {'binary': visit_binary})

    def _annotate_remote_distinct_selectables(self) -> None:
        """annotate 'remote' in primaryjoin, secondaryjoin
        when the parent/child tables are entirely
        separate.

        """

        def repl(element: _CE, **kw: Any) -> Optional[_CE]:
            if self.child_persist_selectable.c.contains_column(element) and (not self.parent_local_selectable.c.contains_column(element) or self.child_local_selectable.c.contains_column(element)):
                return element._annotate({'remote': True})
            return None
        self.primaryjoin = visitors.replacement_traverse(self.primaryjoin, {}, repl)

    def _warn_non_column_elements(self) -> None:
        util.warn('Non-simple column elements in primary join condition for property %s - consider using remote() annotations to mark the remote side.' % self.prop)

    def _annotate_local(self) -> None:
        """Annotate the primaryjoin and secondaryjoin
        structures with 'local' annotations.

        This annotates all column elements found
        simultaneously in the parent table
        and the join condition that don't have a
        'remote' annotation set up from
        _annotate_remote() or user-defined.

        """
        if self._has_annotation(self.primaryjoin, 'local'):
            return
        if self._local_remote_pairs:
            local_side = util.column_set([l for l, r in self._local_remote_pairs])
        else:
            local_side = util.column_set(self.parent_persist_selectable.c)

        def locals_(element: _CE, **kw: Any) -> Optional[_CE]:
            if 'remote' not in element._annotations and element in local_side:
                return element._annotate({'local': True})
            return None
        self.primaryjoin = visitors.replacement_traverse(self.primaryjoin, {}, locals_)

    def _annotate_parentmapper(self) -> None:

        def parentmappers_(element: _CE, **kw: Any) -> Optional[_CE]:
            if 'remote' in element._annotations:
                return element._annotate({'parentmapper': self.prop.mapper})
            elif 'local' in element._annotations:
                return element._annotate({'parentmapper': self.prop.parent})
            return None
        self.primaryjoin = visitors.replacement_traverse(self.primaryjoin, {}, parentmappers_)

    def _check_remote_side(self) -> None:
        if not self.local_remote_pairs:
            raise sa_exc.ArgumentError('Relationship %s could not determine any unambiguous local/remote column pairs based on join condition and remote_side arguments.  Consider using the remote() annotation to accurately mark those elements of the join condition that are on the remote side of the relationship.' % (self.prop,))
        else:
            not_target = util.column_set(self.parent_persist_selectable.c).difference(self.child_persist_selectable.c)
            for _, rmt in self.local_remote_pairs:
                if rmt in not_target:
                    util.warn("Expression %s is marked as 'remote', but these column(s) are local to the local side.  The remote() annotation is needed only for a self-referential relationship where both sides of the relationship refer to the same tables." % (rmt,))

    def _check_foreign_cols(self, join_condition: ColumnElement[bool], primary: bool) -> None:
        """Check the foreign key columns collected and emit error
        messages."""
        can_sync = False
        foreign_cols = self._gather_columns_with_annotation(join_condition, 'foreign')
        has_foreign = bool(foreign_cols)
        if primary:
            can_sync = bool(self.synchronize_pairs)
        else:
            can_sync = bool(self.secondary_synchronize_pairs)
        if self.support_sync and can_sync or (not self.support_sync and has_foreign):
            return
        if self.support_sync and has_foreign and (not can_sync):
            err = "Could not locate any simple equality expressions involving locally mapped foreign key columns for %s join condition '%s' on relationship %s." % (primary and 'primary' or 'secondary', join_condition, self.prop)
            err += "  Ensure that referencing columns are associated with a ForeignKey or ForeignKeyConstraint, or are annotated in the join condition with the foreign() annotation. To allow comparison operators other than '==', the relationship can be marked as viewonly=True."
            raise sa_exc.ArgumentError(err)
        else:
            err = "Could not locate any relevant foreign key columns for %s join condition '%s' on relationship %s." % (primary and 'primary' or 'secondary', join_condition, self.prop)
            err += '  Ensure that referencing columns are associated with a ForeignKey or ForeignKeyConstraint, or are annotated in the join condition with the foreign() annotation.'
            raise sa_exc.ArgumentError(err)

    def _determine_direction(self) -> None:
        """Determine if this relationship is one to many, many to one,
        many to many.

        """
        if self.secondaryjoin is not None:
            self.direction = MANYTOMANY
        else:
            parentcols = util.column_set(self.parent_persist_selectable.c)
            targetcols = util.column_set(self.child_persist_selectable.c)
            onetomany_fk = targetcols.intersection(self.foreign_key_columns)
            manytoone_fk = parentcols.intersection(self.foreign_key_columns)
            if onetomany_fk and manytoone_fk:
                onetomany_local = self._gather_columns_with_annotation(self.primaryjoin, 'remote', 'foreign')
                manytoone_local = {c for c in self._gather_columns_with_annotation(self.primaryjoin, 'foreign') if 'remote' not in c._annotations}
                if onetomany_local and manytoone_local:
                    self_equated = self.remote_columns.intersection(self.local_columns)
                    onetomany_local = onetomany_local.difference(self_equated)
                    manytoone_local = manytoone_local.difference(self_equated)
                if onetomany_local and (not manytoone_local):
                    self.direction = ONETOMANY
                elif manytoone_local and (not onetomany_local):
                    self.direction = MANYTOONE
                else:
                    raise sa_exc.ArgumentError("Can't determine relationship direction for relationship '%s' - foreign key columns within the join condition are present in both the parent and the child's mapped tables.  Ensure that only those columns referring to a parent column are marked as foreign, either via the foreign() annotation or via the foreign_keys argument." % self.prop)
            elif onetomany_fk:
                self.direction = ONETOMANY
            elif manytoone_fk:
                self.direction = MANYTOONE
            else:
                raise sa_exc.ArgumentError("Can't determine relationship direction for relationship '%s' - foreign key columns are present in neither the parent nor the child's mapped tables" % self.prop)

    def _deannotate_pairs(self, collection: _ColumnPairIterable) -> _MutableColumnPairs:
        """provide deannotation for the various lists of
        pairs, so that using them in hashes doesn't incur
        high-overhead __eq__() comparisons against
        original columns mapped.

        """
        return [(x._deannotate(), y._deannotate()) for x, y in collection]

    def _setup_pairs(self) -> None:
        sync_pairs: _MutableColumnPairs = []
        lrp: util.OrderedSet[Tuple[ColumnElement[Any], ColumnElement[Any]]] = util.OrderedSet([])
        secondary_sync_pairs: _MutableColumnPairs = []

        def go(joincond: ColumnElement[bool], collection: _MutableColumnPairs) -> None:

            def visit_binary(binary: BinaryExpression[Any], left: ColumnElement[Any], right: ColumnElement[Any]) -> None:
                if 'remote' in right._annotations and 'remote' not in left._annotations and self.can_be_synced_fn(left):
                    lrp.add((left, right))
                elif 'remote' in left._annotations and 'remote' not in right._annotations and self.can_be_synced_fn(right):
                    lrp.add((right, left))
                if binary.operator is operators.eq and self.can_be_synced_fn(left, right):
                    if 'foreign' in right._annotations:
                        collection.append((left, right))
                    elif 'foreign' in left._annotations:
                        collection.append((right, left))
            visit_binary_product(visit_binary, joincond)
        for joincond, collection in [(self.primaryjoin, sync_pairs), (self.secondaryjoin, secondary_sync_pairs)]:
            if joincond is None:
                continue
            go(joincond, collection)
        self.local_remote_pairs = self._deannotate_pairs(lrp)
        self.synchronize_pairs = self._deannotate_pairs(sync_pairs)
        self.secondary_synchronize_pairs = self._deannotate_pairs(secondary_sync_pairs)
    _track_overlapping_sync_targets: weakref.WeakKeyDictionary[ColumnElement[Any], weakref.WeakKeyDictionary[RelationshipProperty[Any], ColumnElement[Any]]] = weakref.WeakKeyDictionary()

    def _warn_for_conflicting_sync_targets(self) -> None:
        if not self.support_sync:
            return
        for from_, to_ in [(from_, to_) for from_, to_ in self.synchronize_pairs] + [(from_, to_) for from_, to_ in self.secondary_synchronize_pairs]:
            if to_ not in self._track_overlapping_sync_targets:
                self._track_overlapping_sync_targets[to_] = weakref.WeakKeyDictionary({self.prop: from_})
            else:
                other_props = []
                prop_to_from = self._track_overlapping_sync_targets[to_]
                for pr, fr_ in prop_to_from.items():
                    if not pr.mapper._dispose_called and pr not in self.prop._reverse_property and (pr.key not in self.prop._overlaps) and (self.prop.key not in pr._overlaps) and ('__*' not in self.prop._overlaps) and ('__*' not in pr._overlaps) and (not self.prop.parent.is_sibling(pr.parent)) and (not self.prop.mapper.is_sibling(pr.mapper)) and (not self.prop.parent.is_sibling(pr.mapper)) and (not self.prop.mapper.is_sibling(pr.parent)) and (self.prop.key != pr.key or not self.prop.parent.common_parent(pr.parent)):
                        other_props.append((pr, fr_))
                if other_props:
                    util.warn('relationship \'%s\' will copy column %s to column %s, which conflicts with relationship(s): %s. If this is not the intention, consider if these relationships should be linked with back_populates, or if viewonly=True should be applied to one or more if they are read-only. For the less common case that foreign key constraints are partially overlapping, the orm.foreign() annotation can be used to isolate the columns that should be written towards.   To silence this warning, add the parameter \'overlaps="%s"\' to the \'%s\' relationship.' % (self.prop, from_, to_, ', '.join(sorted(("'%s' (copies %s to %s)" % (pr, fr_, to_) for pr, fr_ in other_props))), ','.join(sorted((pr.key for pr, fr in other_props))), self.prop), code='qzyx')
                self._track_overlapping_sync_targets[to_][self.prop] = from_

    @util.memoized_property
    def remote_columns(self) -> Set[ColumnElement[Any]]:
        return self._gather_join_annotations('remote')

    @util.memoized_property
    def local_columns(self) -> Set[ColumnElement[Any]]:
        return self._gather_join_annotations('local')

    @util.memoized_property
    def foreign_key_columns(self) -> Set[ColumnElement[Any]]:
        return self._gather_join_annotations('foreign')

    def _gather_join_annotations(self, annotation: str) -> Set[ColumnElement[Any]]:
        s = set(self._gather_columns_with_annotation(self.primaryjoin, annotation))
        if self.secondaryjoin is not None:
            s.update(self._gather_columns_with_annotation(self.secondaryjoin, annotation))
        return {x._deannotate() for x in s}

    def _gather_columns_with_annotation(self, clause: ColumnElement[Any], *annotation: Iterable[str]) -> Set[ColumnElement[Any]]:
        annotation_set = set(annotation)
        return {cast(ColumnElement[Any], col) for col in visitors.iterate(clause, {}) if annotation_set.issubset(col._annotations)}

    @util.memoized_property
    def _secondary_lineage_set(self) -> FrozenSet[ColumnElement[Any]]:
        if self.secondary is not None:
            return frozenset(itertools.chain(*[c.proxy_set for c in self.secondary.c]))
        else:
            return util.EMPTY_SET

    def join_targets(self, source_selectable: Optional[FromClause], dest_selectable: FromClause, aliased: bool, single_crit: Optional[ColumnElement[bool]]=None, extra_criteria: Tuple[ColumnElement[bool], ...]=()) -> Tuple[ColumnElement[bool], Optional[ColumnElement[bool]], Optional[FromClause], Optional[ClauseAdapter], FromClause]:
        """Given a source and destination selectable, create a
        join between them.

        This takes into account aliasing the join clause
        to reference the appropriate corresponding columns
        in the target objects, as well as the extra child
        criterion, equivalent column sets, etc.

        """
        dest_selectable = _shallow_annotate(dest_selectable, {'no_replacement_traverse': True})
        primaryjoin, secondaryjoin, secondary = (self.primaryjoin, self.secondaryjoin, self.secondary)
        if single_crit is not None:
            if secondaryjoin is not None:
                secondaryjoin = secondaryjoin & single_crit
            else:
                primaryjoin = primaryjoin & single_crit
        if extra_criteria:

            def mark_exclude_cols(elem: SupportsAnnotations, annotations: _AnnotationDict) -> SupportsAnnotations:
                """note unrelated columns in the "extra criteria" as either
                should be adapted or not adapted, even though they are not
                part of our "local" or "remote" side.

                see #9779 for this case, as well as #11010 for a follow up

                """
                parentmapper_for_element = elem._annotations.get('parentmapper', None)
                if parentmapper_for_element is not self.prop.parent and parentmapper_for_element is not self.prop.mapper and (elem not in self._secondary_lineage_set):
                    return _safe_annotate(elem, annotations)
                else:
                    return elem
            extra_criteria = tuple((_deep_annotate(elem, {'should_not_adapt': True}, annotate_callable=mark_exclude_cols) for elem in extra_criteria))
            if secondaryjoin is not None:
                secondaryjoin = secondaryjoin & sql.and_(*extra_criteria)
            else:
                primaryjoin = primaryjoin & sql.and_(*extra_criteria)
        if aliased:
            if secondary is not None:
                secondary = secondary._anonymous_fromclause(flat=True)
                primary_aliasizer = ClauseAdapter(secondary, exclude_fn=_local_col_exclude)
                secondary_aliasizer = ClauseAdapter(dest_selectable, equivalents=self.child_equivalents).chain(primary_aliasizer)
                if source_selectable is not None:
                    primary_aliasizer = ClauseAdapter(secondary, exclude_fn=_local_col_exclude).chain(ClauseAdapter(source_selectable, equivalents=self.parent_equivalents))
                secondaryjoin = secondary_aliasizer.traverse(secondaryjoin)
            else:
                primary_aliasizer = ClauseAdapter(dest_selectable, exclude_fn=_local_col_exclude, equivalents=self.child_equivalents)
                if source_selectable is not None:
                    primary_aliasizer.chain(ClauseAdapter(source_selectable, exclude_fn=_remote_col_exclude, equivalents=self.parent_equivalents))
                secondary_aliasizer = None
            primaryjoin = primary_aliasizer.traverse(primaryjoin)
            target_adapter = secondary_aliasizer or primary_aliasizer
            target_adapter.exclude_fn = None
        else:
            target_adapter = None
        return (primaryjoin, secondaryjoin, secondary, target_adapter, dest_selectable)

    def create_lazy_clause(self, reverse_direction: bool=False) -> Tuple[ColumnElement[bool], Dict[str, ColumnElement[Any]], Dict[ColumnElement[Any], ColumnElement[Any]]]:
        binds: Dict[ColumnElement[Any], BindParameter[Any]] = {}
        equated_columns: Dict[ColumnElement[Any], ColumnElement[Any]] = {}
        has_secondary = self.secondaryjoin is not None
        if has_secondary:
            lookup = collections.defaultdict(list)
            for l, r in self.local_remote_pairs:
                lookup[l].append((l, r))
                equated_columns[r] = l
        elif not reverse_direction:
            for l, r in self.local_remote_pairs:
                equated_columns[r] = l
        else:
            for l, r in self.local_remote_pairs:
                equated_columns[l] = r

        def col_to_bind(element: ColumnElement[Any], **kw: Any) -> Optional[BindParameter[Any]]:
            if not reverse_direction and 'local' in element._annotations or (reverse_direction and (has_secondary and element in lookup or (not has_secondary and 'remote' in element._annotations))):
                if element not in binds:
                    binds[element] = sql.bindparam(None, None, type_=element.type, unique=True)
                return binds[element]
            return None
        lazywhere = self.primaryjoin
        if self.secondaryjoin is None or not reverse_direction:
            lazywhere = visitors.replacement_traverse(lazywhere, {}, col_to_bind)
        if self.secondaryjoin is not None:
            secondaryjoin = self.secondaryjoin
            if reverse_direction:
                secondaryjoin = visitors.replacement_traverse(secondaryjoin, {}, col_to_bind)
            lazywhere = sql.and_(lazywhere, secondaryjoin)
        bind_to_col = {binds[col].key: col for col in binds}
        return (lazywhere, bind_to_col, equated_columns)