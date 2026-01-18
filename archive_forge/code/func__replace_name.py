from inspect import getdoc
from .group import GroupMixin
from .adjoint import AdjointMixin
from .linear import LinearMixin
from .multiply import MultiplyMixin
from .tolerances import TolerancesMixin
def _replace_name(mixin, methods):
    if issubclass(cls, mixin):
        for i in methods:
            meth = getattr(cls, i)
            doc = getdoc(meth)
            if doc is not None:
                meth.__doc__ = doc.replace('CLASS', cls.__name__)