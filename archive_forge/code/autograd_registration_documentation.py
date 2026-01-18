import contextlib
import torch
import torch.utils._pytree as pytree
Check if autograd was registered correctly (for the operator).

    Operators should have "autograd support" registered directly to an
    autograd dispatch key.
    An incorrect registration may lead to unexpected silent incorrectness.
    Note that this check won't catch all problems but will catch
    the most common ones.

    Example usage:
        >>> x = torch.randn(3, requires_grad=True)
        >>> autograd_registration_check(torch.ops.aten.sin.default, (x,), {})

    Here are some best practices if you do find your autograd is
    registered incorrectly:
    - If the operator is composite (i.e. consists of other PyTorch ops)
      and you wish the operator to decompose and get autograd support
      that way, then please register the implementation to
      DispatchKey::CompositeImplicitAutograd
    - If you're adding an autograd formula for the operator, the correct
      thing to do is to register an autograd.Function to
      DispatchKey::Autograd (preferred) or one of the
      DispatchKey::Autograd<BACKEND> keys. It is NOT OK to register
      an autograd.Function to a backend (e.g. CPU/CUDA) key.
    - If your operator is non-differentiable, then you should register
      an implementation to the Autograd key that uses
      AutoDispatchBelowAutograd and re-invokes the operator.

    