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
class UnmappedColumnError(sa_exc.InvalidRequestError):
    """Mapping operation was requested on an unknown column."""