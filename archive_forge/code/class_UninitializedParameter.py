import torch
from torch._C import _disabled_torch_function_impl
from collections import OrderedDict
class UninitializedParameter(UninitializedTensorMixin, Parameter):
    """A parameter that is not initialized.

    Uninitialized Parameters are a a special case of :class:`torch.nn.Parameter`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.nn.Parameter`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.nn.Parameter`.

    The default device or dtype to use when the parameter is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """
    cls_to_become = Parameter

    def __new__(cls, requires_grad=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        data = torch.empty(0, **factory_kwargs)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.requires_grad, self.data.device, self.data.dtype)
            memo[id(self)] = result
            return result