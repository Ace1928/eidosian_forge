from __future__ import annotations
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql.schema import Constraint
from sqlalchemy.sql.schema import ForeignKeyConstraint
from sqlalchemy.sql.schema import Index
from sqlalchemy.sql.schema import UniqueConstraint
from typing_extensions import TypeGuard
from .. import util
from ..util import sqla_compat
@classmethod
def Skip(cls, reason: Union[str, Sequence[str]]) -> ComparisonResult:
    """the constraint cannot be compared for the provided reason(s).

        The message is logged, but the constraints will be otherwise
        considered equal, meaning that no migration command will be
        generated.
        """
    return cls('skip', ', '.join(util.to_list(reason)))