from __future__ import annotations
from typing import Any
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial
 Convert a SymPy expression to a Aesara graph variable.

        The ``dtypes`` and ``broadcastables`` arguments are used to specify the
        data type, dimension, and broadcasting behavior of the Aesara variables
        corresponding to the free symbols in ``expr``. Each is a mapping from
        SymPy symbols to the value of the corresponding argument to
        ``aesara.tensor.var.TensorVariable``.

        See the corresponding `documentation page`__ for more information on
        broadcasting in Aesara.

        .. __: https://aesara.readthedocs.io/en/latest/tutorial/broadcasting.html

        Parameters
        ==========

        expr : sympy.core.expr.Expr
            SymPy expression to print.

        dtypes : dict
            Mapping from SymPy symbols to Aesara datatypes to use when creating
            new Aesara variables for those symbols. Corresponds to the ``dtype``
            argument to ``aesara.tensor.var.TensorVariable``. Defaults to ``'floatX'``
            for symbols not included in the mapping.

        broadcastables : dict
            Mapping from SymPy symbols to the value of the ``broadcastable``
            argument to ``aesara.tensor.var.TensorVariable`` to use when creating Aesara
            variables for those symbols. Defaults to the empty tuple for symbols
            not included in the mapping (resulting in a scalar).

        Returns
        =======

        aesara.graph.basic.Variable
            A variable corresponding to the expression's value in a Aesara
            symbolic expression graph.

        