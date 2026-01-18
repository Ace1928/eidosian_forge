from typing import Optional, Type, Union
from . import counted_lock, errors, lock, transactions, urlutils
from .decorators import only_raises
from .transport import Transport
def _finish_transaction(self):
    """Exit the current transaction."""
    if self._transaction is None:
        raise errors.LockError('Branch %s is not in a transaction' % self)
    transaction = self._transaction
    self._transaction = None
    transaction.finish()