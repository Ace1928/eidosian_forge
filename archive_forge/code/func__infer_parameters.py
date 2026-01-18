import itertools
import warnings
from typing import Protocol
import torch
from ..parameter import is_lazy
def _infer_parameters(self: _LazyProtocol, module, args, kwargs=None):
    """Infers the size and initializes the parameters according to the provided input batch.

        Given a module that contains parameters that were declared inferrable
        using :class:`torch.nn.parameter.ParameterMode.Infer`, runs a forward pass
        in the complete module using the provided input to initialize all the parameters
        as needed.
        The module is set into evaluation mode before running the forward pass in order
        to avoid saving statistics or calculating gradients
        """
    kwargs = kwargs if kwargs else {}
    module.initialize_parameters(*args, **kwargs)
    if module.has_uninitialized_params():
        raise RuntimeError(f'module {self._get_name()} has not been fully initialized')
    module._initialize_hook.remove()
    module._load_hook.remove()
    delattr(module, '_initialize_hook')
    delattr(module, '_load_hook')
    if module.cls_to_become is not None:
        module.__class__ = module.cls_to_become