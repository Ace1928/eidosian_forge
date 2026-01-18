from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import event
from .. import exc
from .. import inspect
from .. import util
from ..orm import PassiveFlag
from ..orm._typing import OrmExecuteOptionsParameter
from ..orm.interfaces import ORMOption
from ..orm.mapper import Mapper
from ..orm.query import Query
from ..orm.session import _BindArguments
from ..orm.session import _PKIdentityArgument
from ..orm.session import Session
from ..util.typing import Protocol
from ..util.typing import Self
class ShardedQuery(Query[_T]):
    """Query class used with :class:`.ShardedSession`.

    .. legacy:: The :class:`.ShardedQuery` is a subclass of the legacy
       :class:`.Query` class.   The :class:`.ShardedSession` now supports
       2.0 style execution via the :meth:`.ShardedSession.execute` method.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.session, ShardedSession)
        self.identity_chooser = self.session.identity_chooser
        self.execute_chooser = self.session.execute_chooser
        self._shard_id = None

    def set_shard(self, shard_id: ShardIdentifier) -> Self:
        """Return a new query, limited to a single shard ID.

        All subsequent operations with the returned query will
        be against the single shard regardless of other state.

        The shard_id can be passed for a 2.0 style execution to the
        bind_arguments dictionary of :meth:`.Session.execute`::

            results = session.execute(
                stmt,
                bind_arguments={"shard_id": "my_shard"}
            )

        """
        return self.execution_options(_sa_shard_id=shard_id)