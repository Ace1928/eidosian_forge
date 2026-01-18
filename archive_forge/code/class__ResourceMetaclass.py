import contextlib
import copy
import weakref
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.trackable import base
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
class _ResourceMetaclass(type):
    """Metaclass for CapturableResource."""

    def __call__(cls, *args, **kwargs):

        def default_resource_creator(next_creator, *a, **kw):
            assert next_creator is None
            obj = cls.__new__(cls, *a, **kw)
            obj.__init__(*a, **kw)
            return obj
        previous_getter = lambda *a, **kw: default_resource_creator(None, *a, **kw)
        resource_creator_stack = ops.get_default_graph()._resource_creator_stack
        for getter in resource_creator_stack[cls._resource_type()]:
            previous_getter = _make_getter(getter, previous_getter)
        return previous_getter(*args, **kwargs)