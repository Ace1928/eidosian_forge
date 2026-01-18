import random
from sympy.core.function import Derivative
from sympy.core.symbol import symbols
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
from sympy.external import import_module
from sympy.functions import \
from sympy.matrices import Matrix, MatrixBase, eye, randMatrix
from sympy.matrices.expressions import \
from sympy.printing.tensorflow import tensorflow_code
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import skip
from sympy.testing.pytest import XFAIL
def _compare_tensorflow_matrix_scalar(variables, expr):
    f = lambdify(variables, expr, 'tensorflow')
    random_matrices = [randMatrix(v.rows, v.cols).evalf() / 100 for v in variables]
    graph = tf.Graph()
    r = None
    with graph.as_default():
        random_variables = [eval(tensorflow_code(i)) for i in random_matrices]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*random_variables))
    e = expr.subs({k: v for k, v in zip(variables, random_matrices)})
    e = e.doit()
    assert abs(r - e) < 10 ** (-6)