from typing import Type
from torch import optim
from .functional_adadelta import _FunctionalAdadelta
from .functional_adagrad import _FunctionalAdagrad
from .functional_adam import _FunctionalAdam
from .functional_adamax import _FunctionalAdamax
from .functional_adamw import _FunctionalAdamW
from .functional_rmsprop import _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop
from .functional_sgd import _FunctionalSGD
def _create_functional_optim(functional_optim_cls: Type, *args, **kwargs):
    return functional_optim_cls([], *args, **kwargs, _allow_empty_param_list=True)