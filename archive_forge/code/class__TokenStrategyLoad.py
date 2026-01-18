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
class _TokenStrategyLoad(_LoadElement):
    """Loader strategies against wildcard attributes

    e.g.::

        raiseload('*')
        Load(User).lazyload('*')
        defer('*')
        load_only(User.name, User.email)  # will create a defer('*')
        joinedload(User.addresses).raiseload('*')

    """
    __visit_name__ = 'token_strategy_load_element'
    inherit_cache = True
    is_class_strategy = False
    is_token_strategy = True

    def _init_path(self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria):
        if attr is not None:
            default_token = attr.endswith(_DEFAULT_TOKEN)
            if attr.endswith(_WILDCARD_TOKEN) or default_token:
                if wildcard_key:
                    attr = f'{wildcard_key}:{attr}'
                path = path.token(attr)
                return path
            else:
                raise sa_exc.ArgumentError('Strings are not accepted for attribute names in loader options; please use class-bound attributes directly.')
        return path

    def _prepare_for_compile_state(self, parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr):
        current_path = compile_state.current_path
        is_refresh = compile_state.compile_options._for_refresh_state
        assert self.path.is_token
        if is_refresh and (not self.propagate_to_loaders):
            return []
        if not self.strategy and (not self.local_opts):
            return []
        effective_path = self.path
        if reconciled_lead_entity:
            effective_path = PathRegistry.coerce((reconciled_lead_entity,) + effective_path.path[1:])
        if current_path:
            new_effective_path = self._adjust_effective_path_for_current_path(effective_path, current_path)
            if new_effective_path is None:
                return []
            effective_path = new_effective_path
        return [('loader', natural_path) for natural_path in cast(TokenRegistry, effective_path)._generate_natural_for_superclasses()]