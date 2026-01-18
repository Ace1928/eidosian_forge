from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter
from ..kvstore import KVStore
def _reset_kvstore(self):
    """Reset kvstore."""
    if self._kvstore and 'dist' in self._kvstore.type:
        raise RuntimeError('Cannot reset distributed KVStore.')
    self._kv_initialized = False
    self._kvstore = None
    self._distributed = None
    self._update_on_kvstore = None
    self._params_to_init = [param for param in self._params]