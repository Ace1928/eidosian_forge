from __future__ import annotations
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from .util import _mapper_property_as_plain_name
from .. import exc as sa_exc
from .. import util
from ..exc import MultipleResultsFound  # noqa
from ..exc import NoResultFound  # noqa
class LoaderStrategyException(sa_exc.InvalidRequestError):
    """A loader strategy for an attribute does not exist."""

    def __init__(self, applied_to_property_type: Type[Any], requesting_property: MapperProperty[Any], applies_to: Optional[Type[MapperProperty[Any]]], actual_strategy_type: Optional[Type[LoaderStrategy]], strategy_key: Tuple[Any, ...]):
        if actual_strategy_type is None:
            sa_exc.InvalidRequestError.__init__(self, "Can't find strategy %s for %s" % (strategy_key, requesting_property))
        else:
            assert applies_to is not None
            sa_exc.InvalidRequestError.__init__(self, 'Can\'t apply "%s" strategy to property "%s", which is a "%s"; this loader strategy is intended to be used with a "%s".' % (util.clsname_as_plain_name(actual_strategy_type), requesting_property, _mapper_property_as_plain_name(applied_to_property_type), _mapper_property_as_plain_name(applies_to)))