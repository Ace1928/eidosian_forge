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
class ObjectDeletedError(sa_exc.InvalidRequestError):
    """A refresh operation failed to retrieve the database
    row corresponding to an object's known primary key identity.

    A refresh operation proceeds when an expired attribute is
    accessed on an object, or when :meth:`_query.Query.get` is
    used to retrieve an object which is, upon retrieval, detected
    as expired.   A SELECT is emitted for the target row
    based on primary key; if no row is returned, this
    exception is raised.

    The true meaning of this exception is simply that
    no row exists for the primary key identifier associated
    with a persistent object.   The row may have been
    deleted, or in some cases the primary key updated
    to a new value, outside of the ORM's management of the target
    object.

    """

    @util.preload_module('sqlalchemy.orm.base')
    def __init__(self, state: InstanceState[Any], msg: Optional[str]=None):
        base = util.preloaded.orm_base
        if not msg:
            msg = "Instance '%s' has been deleted, or its row is otherwise not present." % base.state_str(state)
        sa_exc.InvalidRequestError.__init__(self, msg)

    def __reduce__(self) -> Any:
        return (self.__class__, (None, self.args[0]))