from sympy.core.containers import Tuple
from sympy.core.numbers import oo
from sympy.core.relational import (Gt, Lt)
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import And
from sympy.codegen.ast import (
 Generates an AST for a function implementing the Newton-Raphson method.

    Parameters
    ==========

    expr : expression
    wrt : Symbol
        With respect to, i.e. what is the variable
    params : iterable of symbols
        Symbols appearing in expr that are taken as constants during the iterations
        (these will be accepted as parameters to the generated function).
    func_name : str
        Name of the generated function.
    attrs : Tuple
        Attribute instances passed as ``attrs`` to ``FunctionDefinition``.
    \*\*kwargs :
        Keyword arguments passed to :func:`sympy.codegen.algorithms.newtons_method`.

    Examples
    ========

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.algorithms import newtons_method_function
    >>> from sympy.codegen.pyutils import render_as_module
    >>> x = symbols('x')
    >>> expr = cos(x) - x**3
    >>> func = newtons_method_function(expr, x)
    >>> py_mod = render_as_module(func)  # source code as string
    >>> namespace = {}
    >>> exec(py_mod, namespace, namespace)
    >>> res = eval('newton(0.5)', namespace)
    >>> abs(res - 0.865474033102) < 1e-12
    True

    See Also
    ========

    sympy.codegen.algorithms.newtons_method

    