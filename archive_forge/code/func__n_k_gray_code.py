import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import PauliRot
def _n_k_gray_code(n, k, start=0):
    """Iterates over a full n-ary Gray code with k digits.

    Args:
        n (int): Base of the Gray code. Needs to be greater than one.
        k (int): Number of digits of the Gray code. Needs to be greater than zero.
        start (int, optional): Optional start of the Gray code. The generated code
            will be shorter as the code does not wrap. Defaults to 0.
    """
    for i in range(start, n ** k):
        codeword = [0] * k
        base_repesentation = []
        val = i
        for j in range(k):
            base_repesentation.append(val % n)
            val //= n
        shift = 0
        for j in reversed(range(k)):
            codeword[j] = (base_repesentation[j] + shift) % n
            shift += n - codeword[j]
        yield codeword