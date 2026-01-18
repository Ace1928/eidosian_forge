import contextlib
import functools
import warnings
from typing import Callable, Optional
import torch
from torch._library.utils import Kernel, RegistrationHandle
class AbstractImplCtx:
    """
    Context object for writing abstract implementations for custom operators.
    """

    def __init__(self, _shape_env, _op):
        self._shape_env = _shape_env
        self._op = _op

    def create_unbacked_symint(self, *, min=2, max=None) -> torch.SymInt:
        warnings.warn('create_unbacked_symint is deprecated, please use new_dynamic_size instead')
        return self.new_dynamic_size(min=min, max=max)

    def new_dynamic_size(self, *, min=0, max=None) -> torch.SymInt:
        """Constructs a new symint (symbolic int) representing a data-dependent value.

        This is useful for writing the abstract implementation (which is necessary
        for torch.compile) for a CustomOp where an output Tensor has a size
        that depends on the data of the input Tensors.

        Args:
            min (int): A statically known inclusive lower bound for this symint. Default: 0
            max (Optional[int]): A statically known inclusive upper bound for this
                symint. Default: None

        .. warning:

            It is important that the ``min`` and ``max`` (if not None) values are set
            correctly, otherwise, there will be undefined behavior under
            torch.compile. The default value of ``min`` is 2 due to torch.compile
            specializing on 0/1 sizes.

            You must also verify that your implementation on concrete Tensors
            (e.g. CPU/CUDA) only returns Tensors where the size that corresponds
            to the symint also has respects these constraint.
            The easiest way to do this is to add an assertion in the CPU/CUDA/etc
            implementation that the size follows these bounds.

        Example::

            >>> # An operator with data-dependent output shape
            >>> lib = torch.library.Library("mymodule", "FRAGMENT")
            >>> lib.define("mymodule::custom_nonzero(Tensor x) -> Tensor")
            >>>
            >>> @torch.library.impl_abstract("mymodule::custom_nonzero")
            >>> def custom_nonzero_abstract(x):
            >>>     # Number of nonzero-elements is data-dependent.
            >>>     # Since we cannot peek at the data in an abstract impl,
            >>>     # we use the ctx object to construct a new symint that
            >>>     # represents the data-dependent size.
            >>>     ctx = torch.library.get_ctx()
            >>>     nnz = ctx.new_dynamic_size()
            >>>     shape = [nnz, x.dim()]
            >>>     result = x.new_empty(shape, dtype=torch.int64)
            >>>     return result
            >>>
            >>> @torch.library.impl(lib, "custom_nonzero", "CPU")
            >>> def custom_nonzero_cpu(x):
            >>>     x_np = x.numpy()
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     return torch.tensor(res, device=x.device)

        """
        if self._shape_env is None or not self._shape_env.allow_dynamic_output_shape_ops:
            raise torch._subclasses.fake_tensor.DynamicOutputShapeException(self._op)
        if isinstance(min, torch.SymInt) or isinstance(max, torch.SymInt):
            raise ValueError(f'ctx.new_dynamic_size(min={min}, max={max}): expected min and max to be statically known ints but got SymInt. This is not supported.')
        if min < 0:
            raise ValueError(f'ctx.new_dynamic_size(min={min}, ...): expected min to be greater than or equal to 0: this API can only create non-negative sizes.')
        result = self._shape_env.create_unbacked_symint()
        torch.fx.experimental.symbolic_shapes._constrain_range_for_size(result, min=min, max=max)
        return result