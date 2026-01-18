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
def _compare_tensorflow_scalar(variables, expr, rng=lambda: random.randint(0, 10)):
    f = lambdify(variables, expr, 'tensorflow')
    rvs = [rng() for v in variables]
    graph = tf.Graph()
    r = None
    with graph.as_default():
        tf_rvs = [eval(tensorflow_code(i)) for i in rvs]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*tf_rvs))
    e = expr.subs({k: v for k, v in zip(variables, rvs)}).evalf().doit()
    assert abs(r - e) < 10 ** (-6)