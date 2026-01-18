from typing import Optional, Type, Union
from . import counted_lock, errors, lock, transactions, urlutils
from .decorators import only_raises
from .transport import Transport
def _set_write_transaction(self):
    """Setup a write transaction."""
    self._set_transaction(transactions.WriteTransaction())