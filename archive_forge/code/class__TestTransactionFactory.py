import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
class _TestTransactionFactory(_TransactionFactory):
    """A :class:`._TransactionFactory` used by test suites.

    This is a :class:`._TransactionFactory` that can be directly injected
    with an existing engine and sessionmaker.

    Note that while this is used by oslo.db's own tests of
    the enginefacade system, it is also exported for use by
    the test suites of other projects, first as an element of the
    oslo_db.sqlalchemy.test_fixtures module, and secondly may be used by
    external test suites directly.

    Includes a feature to inject itself temporarily as the factory
    within the global :class:`._TransactionContextManager`.

    """

    @debtcollector.removals.removed_kwarg('synchronous_reader', 'argument value is propagated from the parent _TransactionFactory')
    def __init__(self, engine, maker, apply_global, from_factory=None, **kw):
        self._reader_engine = self._writer_engine = engine
        self._reader_maker = self._writer_maker = maker
        self._started = True
        self._legacy_facade = None
        if from_factory is None:
            from_factory = _context_manager._factory
        self._facade_cfg = from_factory._facade_cfg
        self._transaction_ctx_cfg = from_factory._transaction_ctx_cfg
        self.synchronous_reader = self._facade_cfg['synchronous_reader']
        if apply_global:
            self.existing_factory = _context_manager._factory
            _context_manager._root_factory = self

    def dispose_global(self):
        _context_manager._root_factory = self.existing_factory