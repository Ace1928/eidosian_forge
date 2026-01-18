import contextlib
import functools
import warnings
from typing import Callable, Optional
import torch
from torch._library.utils import Kernel, RegistrationHandle
class AbstractImplHolder:
    """A holder where one can register an abstract impl to."""

    def __init__(self, qualname: str):
        self.qualname: str = qualname
        self.kernel: Optional[Kernel] = None
        self.lib: Optional[torch.library.Library] = None

    def register(self, func: Callable, source: str) -> RegistrationHandle:
        """Register an abstract impl.

        Returns a RegistrationHandle that one can use to de-register this
        abstract impl.
        """
        if self.kernel is not None:
            raise RuntimeError(f'impl_abstract(...): the operator {self.qualname} already has an abstract impl registered at {self.kernel.source}.')
        if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, 'Meta'):
            raise RuntimeError(f"impl_abstract(...): the operator {self.qualname} already has an DispatchKey::Meta implementation via a pre-existing torch.library or TORCH_LIBRARY registration. Please either remove that registration or don't call impl_abstract.")
        if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, 'CompositeImplicitAutograd'):
            raise RuntimeError(f'impl_abstract(...): the operator {self.qualname} already has an implementation for this device type via a pre-existing registration to DispatchKey::CompositeImplicitAutograd.CompositeImplicitAutograd operators do not need an abstract impl; instead, the operator will decompose into its constituents and those can have abstract impls defined on them.')
        self.kernel = Kernel(func, source)
        if self.lib is None:
            ns = self.qualname.split('::')[0]
            self.lib = torch.library.Library(ns, 'FRAGMENT')
        meta_kernel = construct_meta_kernel(self.qualname, self)
        self.lib.impl(self.qualname, meta_kernel, 'Meta')

        def deregister_abstract_impl():
            if self.lib:
                self.lib._destroy()
                self.lib = None
            self.kernel = None
        return RegistrationHandle(deregister_abstract_impl)