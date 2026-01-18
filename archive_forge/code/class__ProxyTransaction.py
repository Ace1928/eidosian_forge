from __future__ import annotations
from contextlib import contextmanager
from contextlib import nullcontext
import logging
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import ContextManager
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import MetaData
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.engine import Engine
from sqlalchemy.engine import url as sqla_url
from sqlalchemy.engine.strategies import MockEngineStrategy
from .. import ddl
from .. import util
from ..util import sqla_compat
from ..util.compat import EncodedIO
class _ProxyTransaction:

    def __init__(self, migration_context: MigrationContext) -> None:
        self.migration_context = migration_context

    @property
    def _proxied_transaction(self) -> Optional[Transaction]:
        return self.migration_context._transaction

    def rollback(self) -> None:
        t = self._proxied_transaction
        assert t is not None
        t.rollback()
        self.migration_context._transaction = None

    def commit(self) -> None:
        t = self._proxied_transaction
        assert t is not None
        t.commit()
        self.migration_context._transaction = None

    def __enter__(self) -> _ProxyTransaction:
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        if self._proxied_transaction is not None:
            self._proxied_transaction.__exit__(type_, value, traceback)
            self.migration_context._transaction = None