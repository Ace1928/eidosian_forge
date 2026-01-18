from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
class FunctionIdentity(serialize.ReduceMixin):
    """
    A function's identity and metadata.

    Note this typically represents a function whose bytecode is
    being compiled, not necessarily the top-level user function
    (the two might be distinct).
    """
    _unique_ids = itertools.count(1)

    @classmethod
    def from_function(cls, pyfunc):
        """
        Create the FunctionIdentity of the given function.
        """
        func = get_function_object(pyfunc)
        code = get_code_object(func)
        pysig = utils.pysignature(func)
        if not code:
            raise errors.ByteCodeSupportError('%s does not provide its bytecode' % func)
        try:
            func_qualname = func.__qualname__
        except AttributeError:
            func_qualname = func.__name__
        self = cls()
        self.func = func
        self.func_qualname = func_qualname
        self.func_name = func_qualname.split('.')[-1]
        self.code = code
        self.module = inspect.getmodule(func)
        self.modname = utils._dynamic_modname if self.module is None else self.module.__name__
        self.is_generator = inspect.isgeneratorfunction(func)
        self.pysig = pysig
        self.filename = code.co_filename
        self.firstlineno = code.co_firstlineno
        self.arg_count = len(pysig.parameters)
        self.arg_names = list(pysig.parameters)
        uid = next(cls._unique_ids)
        self.unique_name = '{}${}'.format(self.func_qualname, uid)
        self.unique_id = uid
        return self

    def derive(self):
        """Copy the object and increment the unique counter.
        """
        return self.from_function(self.func)

    def _reduce_states(self):
        """
        NOTE: part of ReduceMixin protocol
        """
        return dict(pyfunc=self.func)

    @classmethod
    def _rebuild(cls, pyfunc):
        """
        NOTE: part of ReduceMixin protocol
        """
        return cls.from_function(pyfunc)