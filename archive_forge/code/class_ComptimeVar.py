import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
class ComptimeVar:
    """
    A ComptimeVar represents a Python value, at some particular point
    in time, in the Python code we are symbolically evaluating with
    torchdynamo.  This must be distinguished from a runtime value, as
    at compile-time there are some properties of the variable we
    do not know (for example, if the ComptimeVar represents a Tensor,
    we only know metadata about the tensor; we do NOT know what the
    actual data in the Tensor is.)
    """

    def __init__(self, v):
        self.__variable = v

    def as_proxy(self):
        """
        Returns an fx.Proxy (or tuple/list of fx.Proxy) representing
        this variable in the FX graph we are assembling to pass
        to the user compiler.

        This method only works for variables we actually track in
        the FX graph, aka Tensors (and ints, if you are compiling
        with dynamic shapes).  In particular, if you have a list
        or tuple of tensors, you will get a list/tuple of proxies
        (not a single proxy representing the entire list/tuple).
        """
        return self.__variable.as_proxy()

    def is_proxy(self):
        """
        Returns True if as_proxy() would succeed.
        """
        return self.__variable.is_proxy()

    def as_fake(self):
        """
        Returns a "fake" value (either a FakeTensor or a SymInt)
        representing the variable in question.  This only works
        for variables that denote Tensor or int.  You can use
        this to query metadata; e.g., v.as_fake().size(0) will
        tell you the compile-time known size of the tensor.

        WARNING: Do NOT mutate the returned tensor.
        """
        return self.__variable.as_proxy().node.meta['example_value']

    def size(self, dim: Optional[int]=None) -> Union[int, torch.SymInt]:
        """
        Returns the size of the tensor (if dim is None) or the size
        at the dimension dim.  The returned size may be a SymInt.
        """
        return self.as_fake().size(dim)

    def python_type(self):
        """
        Returns what type(v) would have returned for the variable
        at compile time.
        """
        return self.__variable.python_type()

    def as_python_constant(self):
        """
        Returns the Python value this variable would have, but only if it is
        completely known at compile-time (e.g., it is constant).

        WARNING: Do NOT mutate the returned constant.  The returned constant
        may or may not correspond to the actual value this variable may take
        on at runtime; for example, if the variable in question is a constant
        list, we may return a copy of that list.
        """
        return self.__variable.as_python_constant()

    def is_python_constant(self):
        """
        Returns True if as_python_constant would succeed.
        """
        return self.__variable.is_python_constant()

    def _i_will_not_complain_if_bc_breaks_VariableTracker(self):
        """
        Returns the internal data structure VariableTracker that Dynamo uses
        to represent variables at compile time.  There are no BC guarantees on
        this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if you rely on
        it.
        """
        return self.__variable

    def __repr__(self):
        return repr(self.__variable)