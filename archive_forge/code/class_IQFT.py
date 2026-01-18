from sympy.core.expr import Expr
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.functions import sqrt
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError, QExpr
from sympy.matrices import eye
from sympy.physics.quantum.tensorproduct import matrix_tensor_product
from sympy.physics.quantum.gate import (
class IQFT(Fourier):
    """The inverse quantum Fourier transform."""
    gate_name = 'IQFT'
    gate_name_latex = '{QFT^{-1}}'

    def decompose(self):
        """Decomposes IQFT into elementary gates."""
        start = self.args[0]
        finish = self.args[1]
        circuit = 1
        for i in range((finish - start) // 2):
            circuit = SwapGate(i + start, finish - i - 1) * circuit
        for level in range(start, finish):
            for i in reversed(range(level - start)):
                circuit = CGate(level - i - 1, RkGate(level, -i - 2)) * circuit
            circuit = HadamardGate(level) * circuit
        return circuit

    def _eval_inverse(self):
        return QFT(*self.args)

    @property
    def omega(self):
        return exp(-2 * pi * I / self.size)