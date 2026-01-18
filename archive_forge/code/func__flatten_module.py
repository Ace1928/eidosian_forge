import re
from tensorflow.python import tf2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
def _flatten_module(module, recursive, predicate, attribute_traversal_key, attributes_to_ignore, with_path, expand_composites, module_path=(), seen=None, recursion_stack=None):
    """Implementation of `flatten`.

  Args:
    module: Current module to process.
    recursive: Whether to recurse into child modules or not.
    predicate: (Optional) If set then only values matching predicate are
      yielded. A value of `None` (the default) means no items will be
      filtered.
    attribute_traversal_key: (Optional) Method to rekey object attributes
      before they are sorted. Contract is the same as `key` argument to
      builtin `sorted` and only applies to object properties.
    attributes_to_ignore: object attributes to ignored.
    with_path: (Optional) Whether to include the path to the object as well
      as the object itself. If `with_path` is `True` then leaves will not be
      de-duplicated (e.g. if the same leaf instance is reachable via multiple
      modules then it will be yielded multiple times with different paths).
    expand_composites: If true, then composite tensors are expanded into their
      component tensors.
    module_path: The path to the current module as a tuple.
    seen: A set containing all leaf IDs seen so far.
    recursion_stack: A list containing all module IDs associated with the
      current call stack.

  Yields:
    Matched leaves with the optional corresponding paths of the current module
    and optionally all its submodules.
  """
    module_id = id(module)
    if seen is None:
        seen = set([module_id])
    module_dict = vars(module)
    submodules = []
    if recursion_stack is None:
        recursion_stack = []
    if module_id in recursion_stack:
        recursive = False
    for key in sorted(module_dict, key=attribute_traversal_key):
        if key in attributes_to_ignore:
            continue
        prop = module_dict[key]
        try:
            if expand_composites:
                leaves = list(_flatten_non_variable_composites_with_tuple_path(prop))
            else:
                leaves = nest.flatten_with_tuple_paths(prop)
        except Exception as cause:
            raise ValueError('Error processing property {!r} of {!r}'.format(key, prop)) from cause
        for leaf_path, leaf in leaves:
            leaf_path = (key,) + leaf_path
            if not with_path:
                leaf_id = id(leaf)
                if leaf_id in seen:
                    continue
                seen.add(leaf_id)
            if predicate(leaf):
                if with_path:
                    yield (module_path + leaf_path, leaf)
                else:
                    yield leaf
            if recursive and _is_module(leaf):
                submodules.append((module_path + leaf_path, leaf))
    recursion_stack.append(module_id)
    for submodule_path, submodule in submodules:
        subvalues = _flatten_module(submodule, recursive=recursive, predicate=predicate, attribute_traversal_key=attribute_traversal_key, attributes_to_ignore=submodule._TF_MODULE_IGNORED_PROPERTIES, with_path=with_path, expand_composites=expand_composites, module_path=submodule_path, seen=seen, recursion_stack=recursion_stack)
        for subvalue in subvalues:
            yield subvalue
    recursion_stack.pop()