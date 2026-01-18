from __future__ import annotations
import dataclasses
import functools
import inspect
import sys
from collections import OrderedDict, defaultdict, deque, namedtuple
from operator import methodcaller
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Sequence, overload
from typing_extensions import Self  # Python 3.11+
from optree import _C
from optree.typing import (
from optree.utils import safe_zip, total_order_sorted, unzip2
@register_pytree_node_class(namespace=__GLOBAL_NAMESPACE)
class Partial(functools.partial, CustomTreeNode[Any]):
    """A version of :func:`functools.partial` that works in pytrees.

    Use it for partial function evaluation in a way that is compatible with transformations,
    e.g., ``Partial(func, *args, **kwargs)``.

    (You need to explicitly opt-in to this behavior because we did not want to give
    :func:`functools.partial` different semantics than normal function closures.)

    For example, here is a basic usage of :class:`Partial` in a manner similar to
    :func:`functools.partial`:

    >>> import operator
    >>> import torch
    >>> add_one = Partial(operator.add, torch.ones(()))
    >>> add_one(torch.tensor([[1, 2], [3, 4]]))
    tensor([[2., 3.],
            [4., 5.]])

    Pytree compatibility means that the resulting partial function can be passed as an argument
    within tree-map functions, which is not possible with a standard :func:`functools.partial`
    function:

    >>> def call_func_on_cuda(f, *args, **kwargs):
    ...     f, args, kwargs = tree_map(lambda t: t.cuda(), (f, args, kwargs))
    ...     return f(*args, **kwargs)
    ...
    >>> # doctest: +SKIP
    >>> tree_map(lambda t: t.cuda(), add_one)
    Partial(<built-in function add>, tensor(1., device='cuda:0'))
    >>> call_func_on_cuda(add_one, torch.tensor([[1, 2], [3, 4]]))
    tensor([[2., 3.],
            [4., 5.]], device='cuda:0')

    Passing zero arguments to :class:`Partial` effectively wraps the original function, making it a
    valid argument in tree-map functions:

    >>> # doctest: +SKIP
    >>> call_func_on_cuda(Partial(torch.add), torch.tensor(1), torch.tensor(2))
    tensor(3, device='cuda:0')

    Had we passed :func:`operator.add` to ``call_func_on_cuda`` directly, it would have resulted in
    a :class:`TypeError` or :class:`AttributeError`.
    """
    func: Callable[..., Any]
    args: tuple[Any, ...]
    keywords: dict[str, Any]

    def __new__(cls, func: Callable[..., Any], *args: Any, **keywords: Any) -> Self:
        """Create a new :class:`Partial` instance."""
        if isinstance(func, functools.partial):
            original_func = func
            func = _HashablePartialShim(original_func)
            assert not hasattr(func, 'func'), 'shimmed function should not have a `func` attribute'
            out = super().__new__(cls, func, *args, **keywords)
            func.func = original_func.func
            func.args = original_func.args
            func.keywords = original_func.keywords
            return out
        return super().__new__(cls, func, *args, **keywords)

    def tree_flatten(self) -> tuple[tuple[tuple[Any, ...], dict[str, Any]], Callable[..., Any]]:
        """Flatten the :class:`Partial` instance to children and auxiliary data."""
        return ((self.args, self.keywords), self.func)

    @classmethod
    def tree_unflatten(cls, metadata: Callable[..., Any], children: tuple[tuple[Any, ...], dict[str, Any]]) -> Self:
        """Unflatten the children and auxiliary data into a :class:`Partial` instance."""
        args, keywords = children
        return cls(metadata, *args, **keywords)