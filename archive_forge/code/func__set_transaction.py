from typing import Optional, Type, Union
from . import counted_lock, errors, lock, transactions, urlutils
from .decorators import only_raises
from .transport import Transport
def _set_transaction(self, new_transaction):
    """Set a new active transaction."""
    if self._transaction is not None:
        raise errors.LockError('Branch %s is in a transaction already.' % self)
    self._transaction = new_transaction