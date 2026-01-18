from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter
from ..kvstore import KVStore
def _init_kvstore(self):
    """Create kvstore."""
    config = self._kvstore_params
    if self._contains_sparse_weight:
        kvstore, update_on_kvstore = _create_sparse_kvstore(config['kvstore'])
        self._distributed = 'dist' in kvstore.type
        if config['update_on_kvstore'] is False:
            raise ValueError('Cannot set update_on_kvstore=False when sparse weights are present.')
    elif self._contains_sparse_grad:
        arg_arrays = {param.name: param.data(self._contexts[0]) for param in self._params}
        kvstore, _ = _create_kvstore(config['kvstore'], len(self._contexts), arg_arrays)
        self._distributed = 'dist' in kvstore.type if kvstore else False
        update_on_kvstore = self._distributed
        if config['update_on_kvstore'] is not None:
            if config['update_on_kvstore'] is False and self._distributed:
                raise ValueError('Cannot set update_on_kvstore=False on dist kvstore when sparse gradients are present.')
            update_on_kvstore = config['update_on_kvstore']
        if kvstore is not None and (not isinstance(kvstore, KVStore)):
            raise ValueError('Cannot use {} for multi-device training with sparse gradients'.format(type(kvstore)))
    else:
        arg_arrays = {param.name: param.data(self._contexts[0]) for param in self._params}
        kvstore, update_on_kvstore = _create_kvstore(config['kvstore'], len(self._contexts), arg_arrays)
        self._distributed = 'dist' in kvstore.type if kvstore else False
        if self._distributed and 'async' in kvstore.type:
            update_on_kvstore = True
            if config['update_on_kvstore'] is False:
                raise ValueError('Please set update_on_kvstore=True when training in async mode.')
        if config['update_on_kvstore'] is not None:
            update_on_kvstore = config['update_on_kvstore']
        if update_on_kvstore and (not kvstore.is_capable('optimizer')):
            if config['update_on_kvstore']:
                raise ValueError('Please set update_on_kvstore=False when training with {}'.format(type(kvstore)))
            update_on_kvstore = False
    if kvstore:
        if self._compression_params:
            kvstore.set_gradient_compression(self._compression_params)
        if update_on_kvstore:
            kvstore.set_optimizer(self._optimizer)
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
    else:
        self._kvstore = None
        self._update_on_kvstore = None
    self._kv_initialized = True