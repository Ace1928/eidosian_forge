from typing import Optional, Type, Union
from . import counted_lock, errors, lock, transactions, urlutils
from .decorators import only_raises
from .transport import Transport
def _set_read_transaction(self):
    """Setup a read transaction."""
    self._set_transaction(transactions.ReadOnlyTransaction())
    self.get_transaction().set_cache_size(5000)