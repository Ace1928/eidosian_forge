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
def _find_entity_basestring(self, entities: Iterable[_InternalEntityType[Any]], token: str, raiseerr: bool) -> Optional[_InternalEntityType[Any]]:
    if token.endswith(f':{_WILDCARD_TOKEN}'):
        if len(list(entities)) != 1:
            if raiseerr:
                raise sa_exc.ArgumentError(f"Can't apply wildcard ('*') or load_only() loader option to multiple entities {', '.join((str(ent) for ent in entities))}. Specify loader options for each entity individually, such as {', '.join((f"Load({ent}).some_option('*')" for ent in entities))}.")
    elif token.endswith(_DEFAULT_TOKEN):
        raiseerr = False
    for ent in entities:
        return ent
    else:
        if raiseerr:
            raise sa_exc.ArgumentError(f'''Query has only expression-based entities - can't find property named "{token}".''')
        else:
            return None