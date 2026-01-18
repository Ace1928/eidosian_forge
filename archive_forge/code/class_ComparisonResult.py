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
class ComparisonResult(NamedTuple):
    status: Literal['equal', 'different', 'skip']
    message: str

    @property
    def is_equal(self) -> bool:
        return self.status == 'equal'

    @property
    def is_different(self) -> bool:
        return self.status == 'different'

    @property
    def is_skip(self) -> bool:
        return self.status == 'skip'

    @classmethod
    def Equal(cls) -> ComparisonResult:
        """the constraints are equal."""
        return cls('equal', 'The two constraints are equal')

    @classmethod
    def Different(cls, reason: Union[str, Sequence[str]]) -> ComparisonResult:
        """the constraints are different for the provided reason(s)."""
        return cls('different', ', '.join(util.to_list(reason)))

    @classmethod
    def Skip(cls, reason: Union[str, Sequence[str]]) -> ComparisonResult:
        """the constraint cannot be compared for the provided reason(s).

        The message is logged, but the constraints will be otherwise
        considered equal, meaning that no migration command will be
        generated.
        """
        return cls('skip', ', '.join(util.to_list(reason)))