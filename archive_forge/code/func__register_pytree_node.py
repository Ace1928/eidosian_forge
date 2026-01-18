import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def _register_pytree_node(cls: Type[Any], flatten_fn: FlattenFunc, unflatten_fn: UnflattenFunc, *, serialized_type_name: Optional[str]=None, to_dumpable_context: Optional[ToDumpableContextFn]=None, from_dumpable_context: Optional[FromDumpableContextFn]=None) -> None:
    """Register a container-like type as pytree node for the C++ pytree only.

    The ``namespace`` argument is used to avoid collisions that occur when different libraries
    register the same Python type with different behaviors. It is recommended to add a unique prefix
    to the namespace to avoid conflicts with other libraries. Namespaces can also be used to specify
    the same class in different namespaces for different use cases.

    .. warning::
        For safety reasons, a ``namespace`` must be specified while registering a custom type. It is
        used to isolate the behavior of flattening and unflattening a pytree node type. This is to
        prevent accidental collisions between different libraries that may register the same type.

    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_fn (callable): A function to be used during flattening, taking an instance of
            ``cls`` and returning a pair, with (1) an iterable for the children to be flattened
            recursively, and (2) some hashable auxiliary data to be stored in the treespec and to be
            passed to the ``unflatten_fn``.
        unflatten_fn (callable): A function taking two arguments: the auxiliary data that was
            returned by ``flatten_fn`` and stored in the treespec, and the unflattened children.
            The function should return an instance of ``cls``.
        serialized_type_name (str, optional): A keyword argument used to specify the fully
            qualified name used when serializing the tree spec.
        to_dumpable_context (callable, optional): An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable representation. This is
            used for json serialization, which is being used in :mod:`torch.export` right now.
        from_dumpable_context (callable, optional): An optional keyword argument to custom specify
            how to convert the custom json dumpable representation of the context back to the
            original context. This is used for json deserialization, which is being used in
            :mod:`torch.export` right now.
    """
    warnings.warn('torch.utils._cxx_pytree._register_pytree_node is deprecated. Please use torch.utils._cxx_pytree.register_pytree_node instead.', stacklevel=2)
    _private_register_pytree_node(cls, flatten_fn, unflatten_fn, serialized_type_name=serialized_type_name, to_dumpable_context=to_dumpable_context, from_dumpable_context=from_dumpable_context)