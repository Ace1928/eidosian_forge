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
class StaleDataError(sa_exc.SQLAlchemyError):
    """An operation encountered database state that is unaccounted for.

    Conditions which cause this to happen include:

    * A flush may have attempted to update or delete rows
      and an unexpected number of rows were matched during
      the UPDATE or DELETE statement.   Note that when
      version_id_col is used, rows in UPDATE or DELETE statements
      are also matched against the current known version
      identifier.

    * A mapped object with version_id_col was refreshed,
      and the version number coming back from the database does
      not match that of the object itself.

    * A object is detached from its parent object, however
      the object was previously attached to a different parent
      identity which was garbage collected, and a decision
      cannot be made if the new parent was really the most
      recent "parent".

    """