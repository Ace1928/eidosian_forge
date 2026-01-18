from functools import wraps
from sympy.utilities.decorator import threaded, xthreaded, memoize_property, deprecated
from sympy.testing.pytest import warns_deprecated_sympy
from sympy.core.basic import Basic
from sympy.core.relational import Eq
from sympy.matrices.dense import Matrix
from sympy.abc import x, y
@deprecated('deprecated_class is deprecated', deprecated_since_version='1.10', active_deprecations_target='active-deprecations')
class deprecated_class:
    pass