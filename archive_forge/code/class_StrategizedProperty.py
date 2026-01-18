from __future__ import annotations
import collections
import dataclasses
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc as orm_exc
from . import path_registry
from .base import _MappedAttribute as _MappedAttribute
from .base import EXT_CONTINUE as EXT_CONTINUE  # noqa: F401
from .base import EXT_SKIP as EXT_SKIP  # noqa: F401
from .base import EXT_STOP as EXT_STOP  # noqa: F401
from .base import InspectionAttr as InspectionAttr  # noqa: F401
from .base import InspectionAttrInfo as InspectionAttrInfo
from .base import MANYTOMANY as MANYTOMANY  # noqa: F401
from .base import MANYTOONE as MANYTOONE  # noqa: F401
from .base import NO_KEY as NO_KEY  # noqa: F401
from .base import NO_VALUE as NO_VALUE  # noqa: F401
from .base import NotExtension as NotExtension  # noqa: F401
from .base import ONETOMANY as ONETOMANY  # noqa: F401
from .base import RelationshipDirection as RelationshipDirection  # noqa: F401
from .base import SQLORMOperations
from .. import ColumnElement
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import ExecutableOption
from ..sql.cache_key import HasCacheKey
from ..sql.operators import ColumnOperators
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import warn_deprecated
from ..util.typing import RODescriptorReference
from ..util.typing import TypedDict
class StrategizedProperty(MapperProperty[_T]):
    """A MapperProperty which uses selectable strategies to affect
    loading behavior.

    There is a single strategy selected by default.  Alternate
    strategies can be selected at Query time through the usage of
    ``StrategizedOption`` objects via the Query.options() method.

    The mechanics of StrategizedProperty are used for every Query
    invocation for every mapped attribute participating in that Query,
    to determine first how the attribute will be rendered in SQL
    and secondly how the attribute will retrieve a value from a result
    row and apply it to a mapped object.  The routines here are very
    performance-critical.

    """
    __slots__ = ('_strategies', 'strategy', '_wildcard_token', '_default_path_loader_key', 'strategy_key')
    inherit_cache = True
    strategy_wildcard_key: ClassVar[str]
    strategy_key: _StrategyKey
    _strategies: Dict[_StrategyKey, LoaderStrategy]

    def _memoized_attr__wildcard_token(self) -> Tuple[str]:
        return (f'{self.strategy_wildcard_key}:{path_registry._WILDCARD_TOKEN}',)

    def _memoized_attr__default_path_loader_key(self) -> Tuple[str, Tuple[str]]:
        return ('loader', (f'{self.strategy_wildcard_key}:{path_registry._DEFAULT_TOKEN}',))

    def _get_context_loader(self, context: ORMCompileState, path: AbstractEntityRegistry) -> Optional[_LoadElement]:
        load: Optional[_LoadElement] = None
        search_path = path[self]
        for path_key in (search_path._loader_key, search_path._wildcard_path_loader_key, search_path._default_path_loader_key):
            if path_key in context.attributes:
                load = context.attributes[path_key]
                break
        return load

    def _get_strategy(self, key: _StrategyKey) -> LoaderStrategy:
        try:
            return self._strategies[key]
        except KeyError:
            pass
        cls = self._strategy_lookup(self, *key)
        self._strategies[key] = strategy = cls(self, key)
        return strategy

    def setup(self, context: ORMCompileState, query_entity: _MapperEntity, path: AbstractEntityRegistry, adapter: Optional[ORMAdapter], **kwargs: Any) -> None:
        loader = self._get_context_loader(context, path)
        if loader and loader.strategy:
            strat = self._get_strategy(loader.strategy)
        else:
            strat = self.strategy
        strat.setup_query(context, query_entity, path, loader, adapter, **kwargs)

    def create_row_processor(self, context: ORMCompileState, query_entity: _MapperEntity, path: AbstractEntityRegistry, mapper: Mapper[Any], result: Result[Any], adapter: Optional[ORMAdapter], populators: _PopulatorDict) -> None:
        loader = self._get_context_loader(context, path)
        if loader and loader.strategy:
            strat = self._get_strategy(loader.strategy)
        else:
            strat = self.strategy
        strat.create_row_processor(context, query_entity, path, loader, mapper, result, adapter, populators)

    def do_init(self) -> None:
        self._strategies = {}
        self.strategy = self._get_strategy(self.strategy_key)

    def post_instrument_class(self, mapper: Mapper[Any]) -> None:
        if not self.parent.non_primary and (not mapper.class_manager._attr_has_impl(self.key)):
            self.strategy.init_class_attribute(mapper)
    _all_strategies: collections.defaultdict[Type[MapperProperty[Any]], Dict[_StrategyKey, Type[LoaderStrategy]]] = collections.defaultdict(dict)

    @classmethod
    def strategy_for(cls, **kw: Any) -> Callable[[_TLS], _TLS]:

        def decorate(dec_cls: _TLS) -> _TLS:
            if '_strategy_keys' not in dec_cls.__dict__:
                dec_cls._strategy_keys = []
            key = tuple(sorted(kw.items()))
            cls._all_strategies[cls][key] = dec_cls
            dec_cls._strategy_keys.append(key)
            return dec_cls
        return decorate

    @classmethod
    def _strategy_lookup(cls, requesting_property: MapperProperty[Any], *key: Any) -> Type[LoaderStrategy]:
        requesting_property.parent._with_polymorphic_mappers
        for prop_cls in cls.__mro__:
            if prop_cls in cls._all_strategies:
                if TYPE_CHECKING:
                    assert issubclass(prop_cls, MapperProperty)
                strategies = cls._all_strategies[prop_cls]
                try:
                    return strategies[key]
                except KeyError:
                    pass
        for property_type, strats in cls._all_strategies.items():
            if key in strats:
                intended_property_type = property_type
                actual_strategy = strats[key]
                break
        else:
            intended_property_type = None
            actual_strategy = None
        raise orm_exc.LoaderStrategyException(cls, requesting_property, intended_property_type, actual_strategy, key)