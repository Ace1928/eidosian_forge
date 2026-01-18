from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import util as orm_util
from ._typing import insp_is_aliased_class
from ._typing import insp_is_attribute
from ._typing import insp_is_mapper
from ._typing import insp_is_mapper_property
from .attributes import QueryableAttribute
from .base import InspectionAttr
from .interfaces import LoaderOption
from .path_registry import _DEFAULT_TOKEN
from .path_registry import _StrPathToken
from .path_registry import _WILDCARD_TOKEN
from .path_registry import AbstractEntityRegistry
from .path_registry import path_is_property
from .path_registry import PathRegistry
from .path_registry import TokenRegistry
from .util import _orm_full_deannotate
from .util import AliasedInsp
from .. import exc as sa_exc
from .. import inspect
from .. import util
from ..sql import and_
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import traversals
from ..sql import visitors
from ..sql.base import _generative
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Self
class _LoadElement(cache_key.HasCacheKey, traversals.HasShallowCopy, visitors.Traversible):
    """represents strategy information to select for a LoaderStrategy
    and pass options to it.

    :class:`._LoadElement` objects provide the inner datastructure
    stored by a :class:`_orm.Load` object and are also the object passed
    to methods like :meth:`.LoaderStrategy.setup_query`.

    .. versionadded:: 2.0

    """
    __slots__ = ('path', 'strategy', 'propagate_to_loaders', 'local_opts', '_extra_criteria', '_reconcile_to_other')
    __visit_name__ = 'load_element'
    _traverse_internals = [('path', visitors.ExtendedInternalTraversal.dp_has_cache_key), ('strategy', visitors.ExtendedInternalTraversal.dp_plain_obj), ('local_opts', visitors.ExtendedInternalTraversal.dp_string_multi_dict), ('_extra_criteria', visitors.InternalTraversal.dp_clauseelement_list), ('propagate_to_loaders', visitors.InternalTraversal.dp_plain_obj), ('_reconcile_to_other', visitors.InternalTraversal.dp_plain_obj)]
    _cache_key_traversal = None
    _extra_criteria: Tuple[Any, ...]
    _reconcile_to_other: Optional[bool]
    strategy: Optional[_StrategyKey]
    path: PathRegistry
    propagate_to_loaders: bool
    local_opts: util.immutabledict[str, Any]
    is_token_strategy: bool
    is_class_strategy: bool

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other):
        return traversals.compare(self, other)

    @property
    def is_opts_only(self) -> bool:
        return bool(self.local_opts and self.strategy is None)

    def _clone(self, **kw: Any) -> _LoadElement:
        cls = self.__class__
        s = cls.__new__(cls)
        self._shallow_copy_to(s)
        return s

    def _update_opts(self, **kw: Any) -> _LoadElement:
        new = self._clone()
        new.local_opts = new.local_opts.union(kw)
        return new

    def __getstate__(self) -> Dict[str, Any]:
        d = self._shallow_to_dict()
        d['path'] = self.path.serialize()
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        state['path'] = PathRegistry.deserialize(state['path'])
        self._shallow_from_dict(state)

    def _raise_for_no_match(self, parent_loader, mapper_entities):
        path = parent_loader.path
        found_entities = False
        for ent in mapper_entities:
            ezero = ent.entity_zero
            if ezero:
                found_entities = True
                break
        if not found_entities:
            raise sa_exc.ArgumentError(f"Query has only expression-based entities; attribute loader options for {path[0]} can't be applied here.")
        else:
            raise sa_exc.ArgumentError(f'Mapped class {path[0]} does not apply to any of the root entities in this query, e.g. {', '.join((str(x.entity_zero) for x in mapper_entities if x.entity_zero))}. Please specify the full path from one of the root entities to the target attribute. ')

    def _adjust_effective_path_for_current_path(self, effective_path: PathRegistry, current_path: PathRegistry) -> Optional[PathRegistry]:
        """receives the 'current_path' entry from an :class:`.ORMCompileState`
        instance, which is set during lazy loads and secondary loader strategy
        loads, and adjusts the given path to be relative to the
        current_path.

        E.g. given a loader path and current path::

            lp: User -> orders -> Order -> items -> Item -> keywords -> Keyword

            cp: User -> orders -> Order -> items

        The adjusted path would be::

            Item -> keywords -> Keyword


        """
        chopped_start_path = Load._chop_path(effective_path.natural_path, current_path)
        if not chopped_start_path:
            return None
        tokens_removed_from_start_path = len(effective_path) - len(chopped_start_path)
        loader_lead_path_element = self.path[tokens_removed_from_start_path]
        effective_path = PathRegistry.coerce((loader_lead_path_element,) + chopped_start_path[1:])
        return effective_path

    def _init_path(self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria):
        """Apply ORM attributes and/or wildcard to an existing path, producing
        a new path.

        This method is used within the :meth:`.create` method to initialize
        a :class:`._LoadElement` object.

        """
        raise NotImplementedError()

    def _prepare_for_compile_state(self, parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr):
        """implemented by subclasses."""
        raise NotImplementedError()

    def process_compile_state(self, parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr):
        """populate ORMCompileState.attributes with loader state for this
        _LoadElement.

        """
        keys = self._prepare_for_compile_state(parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr)
        for key in keys:
            if key in compile_state.attributes:
                compile_state.attributes[key] = _LoadElement._reconcile(self, compile_state.attributes[key])
            else:
                compile_state.attributes[key] = self

    @classmethod
    def create(cls, path: PathRegistry, attr: Union[_AttrType, _StrPathToken, None], strategy: Optional[_StrategyKey], wildcard_key: Optional[_WildcardKeyType], local_opts: Optional[_OptsType], propagate_to_loaders: bool, raiseerr: bool=True, attr_group: Optional[_AttrGroupType]=None, reconcile_to_other: Optional[bool]=None, extra_criteria: Optional[Tuple[Any, ...]]=None) -> _LoadElement:
        """Create a new :class:`._LoadElement` object."""
        opt = cls.__new__(cls)
        opt.path = path
        opt.strategy = strategy
        opt.propagate_to_loaders = propagate_to_loaders
        opt.local_opts = util.immutabledict(local_opts) if local_opts else util.EMPTY_DICT
        opt._extra_criteria = ()
        if reconcile_to_other is not None:
            opt._reconcile_to_other = reconcile_to_other
        elif strategy is None and (not local_opts):
            opt._reconcile_to_other = True
        else:
            opt._reconcile_to_other = None
        path = opt._init_path(path, attr, wildcard_key, attr_group, raiseerr, extra_criteria)
        if not path:
            return None
        assert opt.is_token_strategy == path.is_token
        opt.path = path
        return opt

    def __init__(self) -> None:
        raise NotImplementedError()

    def _recurse(self) -> _LoadElement:
        cloned = self._clone()
        cloned.path = PathRegistry.coerce(self.path[:] + self.path[-2:])
        return cloned

    def _prepend_path_from(self, parent: Load) -> _LoadElement:
        """adjust the path of this :class:`._LoadElement` to be
        a subpath of that of the given parent :class:`_orm.Load` object's
        path.

        This is used by the :meth:`_orm.Load._apply_to_parent` method,
        which is in turn part of the :meth:`_orm.Load.options` method.

        """
        if not any((orm_util._entity_corresponds_to_use_path_impl(elem, self.path.odd_element(0)) for elem in (parent.path.odd_element(-1),) + parent.additional_source_entities)):
            raise sa_exc.ArgumentError(f'Attribute "{self.path[1]}" does not link from element "{parent.path[-1]}".')
        return self._prepend_path(parent.path)

    def _prepend_path(self, path: PathRegistry) -> _LoadElement:
        cloned = self._clone()
        assert cloned.strategy == self.strategy
        assert cloned.local_opts == self.local_opts
        assert cloned.is_class_strategy == self.is_class_strategy
        cloned.path = PathRegistry.coerce(path[0:-1] + cloned.path[:])
        return cloned

    @staticmethod
    def _reconcile(replacement: _LoadElement, existing: _LoadElement) -> _LoadElement:
        """define behavior for when two Load objects are to be put into
        the context.attributes under the same key.

        :param replacement: ``_LoadElement`` that seeks to replace the
         existing one

        :param existing: ``_LoadElement`` that is already present.

        """
        if replacement._reconcile_to_other:
            return existing
        elif replacement._reconcile_to_other is False:
            return replacement
        elif existing._reconcile_to_other:
            return replacement
        elif existing._reconcile_to_other is False:
            return existing
        if existing is replacement:
            return replacement
        elif existing.strategy == replacement.strategy and existing.local_opts == replacement.local_opts:
            return replacement
        elif replacement.is_opts_only:
            existing = existing._clone()
            existing.local_opts = existing.local_opts.union(replacement.local_opts)
            existing._extra_criteria += replacement._extra_criteria
            return existing
        elif existing.is_opts_only:
            replacement = replacement._clone()
            replacement.local_opts = replacement.local_opts.union(existing.local_opts)
            replacement._extra_criteria += existing._extra_criteria
            return replacement
        elif replacement.path.is_token:
            return replacement
        raise sa_exc.InvalidRequestError(f'Loader strategies for {replacement.path} conflict')