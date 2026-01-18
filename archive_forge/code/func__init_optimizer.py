from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter
from ..kvstore import KVStore
def _init_optimizer(self, optimizer, optimizer_params):
    param_dict = {i: param for i, param in enumerate(self._params)}
    if isinstance(optimizer, opt.Optimizer):
        assert not optimizer_params, 'optimizer_params must be None if optimizer is an instance of Optimizer instead of str'
        self._optimizer = optimizer
        self._optimizer.param_dict = param_dict
    else:
        self._optimizer = opt.create(optimizer, param_dict=param_dict, **optimizer_params)
    self._updaters = [opt.get_updater(self._optimizer) for _ in self._contexts]