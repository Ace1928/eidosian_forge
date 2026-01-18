from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class ClassType(Callable, Opaque):
    """
    The type of the jitted class (not instance).  When the type of a class
    is called, its constructor is invoked.
    """
    mutable = True
    name_prefix = 'jitclass'
    instance_type_class = ClassInstanceType

    def __init__(self, class_def, ctor_template_cls, struct, jit_methods, jit_props, jit_static_methods):
        self.class_name = class_def.__name__
        self.class_doc = class_def.__doc__
        self._ctor_template_class = ctor_template_cls
        self.jit_methods = jit_methods
        self.jit_props = jit_props
        self.jit_static_methods = jit_static_methods
        self.struct = struct
        fielddesc = ','.join(('{0}:{1}'.format(k, v) for k, v in struct.items()))
        name = '{0}.{1}#{2:x}<{3}>'.format(self.name_prefix, self.class_name, id(self), fielddesc)
        super(ClassType, self).__init__(name)

    def get_call_type(self, context, args, kws):
        return self.ctor_template(context).apply(args, kws)

    def get_call_signatures(self):
        return ((), True)

    def get_impl_key(self, sig):
        return type(self)

    @property
    def methods(self):
        return {k: v.py_func for k, v in self.jit_methods.items()}

    @property
    def static_methods(self):
        return {k: v.py_func for k, v in self.jit_static_methods.items()}

    @property
    def instance_type(self):
        return ClassInstanceType(self)

    @property
    def ctor_template(self):
        return self._specialize_template(self._ctor_template_class)

    def _specialize_template(self, basecls):
        return type(basecls.__name__, (basecls,), dict(key=self))