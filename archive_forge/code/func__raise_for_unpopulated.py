from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import base
from .collections import collection
from .collections import collection_adapter
from .. import exc as sa_exc
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..util.typing import Literal
@util.preload_module('sqlalchemy.orm.attributes')
def _raise_for_unpopulated(self, value: _KT, initiator: Union[AttributeEventToken, Literal[None, False]]=None, *, warn_only: bool) -> None:
    mapper = base.instance_state(value).mapper
    attributes = util.preloaded.orm_attributes
    if not isinstance(initiator, attributes.AttributeEventToken):
        relationship = 'unknown relationship'
    elif initiator.key in mapper.attrs:
        relationship = f'{mapper.attrs[initiator.key]}'
    else:
        relationship = initiator.key
    if warn_only:
        util.warn(f"""Attribute keyed dictionary value for attribute '{relationship}' was None; this will raise in a future release. To skip this assignment entirely, Set the "ignore_unpopulated_attribute=True" parameter on the mapped collection factory.""")
    else:
        raise sa_exc.InvalidRequestError(f"""In event triggered from population of attribute '{relationship}' (potentially from a backref), can't populate value in KeyFuncDict; dictionary key derived from {base.instance_str(value)} is not populated. Ensure appropriate state is set up on the {base.instance_str(value)} object before assigning to the {relationship} attribute. To skip this assignment entirely, Set the "ignore_unpopulated_attribute=True" parameter on the mapped collection factory.""")