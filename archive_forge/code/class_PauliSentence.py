import warnings
from copy import copy
from functools import reduce, lru_cache
from typing import Iterable
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import math
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum
class PauliSentence(dict):
    """Dictionary representing a linear combination of Pauli words, with the keys
    as :class:`~pennylane.pauli.PauliWord` instances and the values correspond to coefficients.

    .. note::

        An empty :class:`~.PauliSentence` will be treated as the additive
        identity (i.e ``0 * Identity()``). Its matrix is the all-zero matrix
        (trivially the :math:`1\\times 1` zero matrix when no ``wire_order`` is passed to
        ``PauliSentence({}).to_mat()``).

    **Examples**

    >>> ps = PauliSentence({
            PauliWord({0:'X', 1:'Y'}): 1.23,
            PauliWord({2:'Z', 0:'Y'}): -0.45j
        })
    >>> ps
    1.23 * X(0) @ Y(1)
    + (-0-0.45j) * Z(2) @ Y(0)

    Combining Pauli words automatically results in Pauli sentences that can be used to construct more complicated operators.

    >>> w1 = PauliWord({0:"X", 1:"Y"})
    >>> w2 = PauliWord({1:"X", 2:"Z"})
    >>> ps = 0.5 * w1 - 1.5 * w2 + 2
    >>> ps + PauliWord({3:"Z"}) - 1
    0.5 * X(0) @ Y(1)
    + -1.5 * X(1) @ Z(2)
    + 1 * I
    + 1.0 * Z(3)

    Note that while the empty :class:`~PauliWord` ``PauliWord({})`` respresents the identity, the empty ``PauliSentence`` represents 0

    >>> PauliSentence({})
    0 * I

    We can compute commutators using the ``PauliSentence.commutator()`` method

    >>> op1 = PauliWord({0:"X", 1:"X"})
    >>> op2 = PauliWord({0:"Y"}) + PauliWord({1:"Y"})
    >>> op1.commutator(op2)
    2j * Z(0) @ X(1)
    + 2j * X(0) @ Z(1)

    Or, alternatively, use :func:`~commutator`.

    >>> qml.commutator(op1, op2, pauli=True)

    Note that we need to specify ``pauli=True`` as :func:`~.commutator` returns PennyLane operators by default.

    """
    __array_priority__ = 1000

    @property
    def pauli_rep(self):
        """Trivial pauli_rep"""
        return self

    def __missing__(self, key):
        """If the PauliWord is not in the sentence then the coefficient
        associated with it should be 0."""
        return 0.0

    def __add__(self, other):
        """Add a PauliWord, scalar or other PauliSentence to a PauliSentence.

        Empty Pauli sentences are treated as the additive identity
        (i.e 0 * Identity on all wires). The non-empty Pauli sentence is returned.
        """
        if isinstance(other, PauliSentence):
            smaller_ps, larger_ps = (self, copy(other)) if len(self) < len(other) else (other, copy(self))
            for key in smaller_ps:
                larger_ps[key] += smaller_ps[key]
            return larger_ps
        if isinstance(other, PauliWord):
            res = copy(self)
            if other in res:
                res[other] += 1.0
            else:
                res[other] = 1.0
            return res
        if isinstance(other, TensorLike):
            res = copy(self)
            IdWord = PauliWord({})
            if IdWord in res:
                res[IdWord] += other
            else:
                res[IdWord] = other
            return res
        raise TypeError(f'Cannot add {other} of type {type(other)} to PauliSentence')
    __radd__ = __add__

    def __iadd__(self, other):
        """Inplace addition of two Pauli sentence together by adding terms of other to self"""
        if isinstance(other, PauliSentence):
            for key in other:
                if key in self:
                    self[key] += other[key]
                else:
                    self[key] = other[key]
            return self
        if isinstance(other, PauliWord):
            if other in self:
                self[other] += 1.0
            else:
                self[other] = 1.0
            return self
        if isinstance(other, TensorLike):
            IdWord = PauliWord({})
            if IdWord in self:
                self[IdWord] += other
            else:
                self[IdWord] = other
            return self
        raise TypeError(f'Cannot add {other} of type {type(other)} to PauliSentence')

    def __sub__(self, other):
        """Subtract other PauliSentence, PauliWord, or scalar"""
        return self + -1 * other

    def __rsub__(self, other):
        """Subtract other PauliSentence, PauliWord, or scalar"""
        return -1 * self + other

    def __copy__(self):
        """Copy the PauliSentence instance."""
        copied_ps = {}
        for pw, coeff in self.items():
            copied_ps[copy(pw)] = coeff
        return PauliSentence(copied_ps)

    def __deepcopy__(self, memo):
        res = self.__copy__()
        memo[id(self)] = res
        return res

    def __matmul__(self, other):
        """Matrix / tensor product between two PauliSentences by iterating over each sentence and multiplying
        the Pauli words pair-wise"""
        if isinstance(other, PauliWord):
            other = PauliSentence({other: 1.0})
        final_ps = PauliSentence()
        if len(self) == 0 or len(other) == 0:
            return final_ps
        for pw1 in self:
            for pw2 in other:
                prod_pw, coeff = pw1._matmul(pw2)
                final_ps[prod_pw] = final_ps[prod_pw] + coeff * self[pw1] * other[pw2]
        return final_ps

    def __mul__(self, other):
        """Multiply a PauliWord by a scalar

        Args:
            other (Scalar): The scalar to multiply the PauliWord with

        Returns:
            PauliSentence
        """
        if isinstance(other, PauliSentence):
            warnings.warn('Matrix/Tensor multiplication using the * operator on PauliWords and PauliSentences is deprecated, use @ instead.', qml.PennyLaneDeprecationWarning)
            return self @ other
        if isinstance(other, TensorLike):
            if not qml.math.ndim(other) == 0:
                raise ValueError(f'Attempting to multiply a PauliSentence with an array of dimension {qml.math.ndim(other)}')
            return PauliSentence({key: other * value for key, value in self.items()})
        raise TypeError(f'PauliSentence can only be multiplied by numerical data. Attempting to multiply by {other} of type {type(other)}')
    __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide a PauliSentence by a scalar"""
        if isinstance(other, TensorLike):
            return self * (1 / other)
        raise TypeError(f'PauliSentence can only be divided by numerical data. Attempting to divide by {other} of type {type(other)}')

    def commutator(self, other):
        """
        Compute commutator between a ``PauliSentence`` :math:`P` and other operator :math:`O`

        .. math:: [P, O] = P O - O P

        When the other operator is a :class:`~PauliWord` or :class:`~PauliSentence`,
        this method is faster than computing ``P @ O - O @ P``. It is what is being used
        in :func:`~commutator` when setting ``pauli=True``.

        Args:
            other (Union[Operator, PauliWord, PauliSentence]): Second operator

        Returns:
            ~PauliSentence: The commutator result in form of a :class:`~PauliSentence` instances.

        **Examples**

        You can compute commutators between :class:`~PauliSentence` instances.

        >>> pw1 = PauliWord({0:"X"})
        >>> pw2 = PauliWord({1:"X"})
        >>> ps1 = PauliSentence({pw1: 1., pw2: 2.})
        >>> ps2 = PauliSentence({pw1: 0.5j, pw2: 1j})
        >>> ps1.commutator(ps2)
        0 * I

        You can also compute the commutator with other operator types if they have a Pauli representation.

        >>> ps1.commutator(qml.Y(0))
        2j * Z(0)"""
        final_ps = PauliSentence()
        if isinstance(other, PauliWord):
            for pw1 in self:
                comm_pw, coeff = pw1._commutator(other)
                if len(comm_pw) != 0:
                    final_ps[comm_pw] += coeff * self[pw1]
            return final_ps
        if not isinstance(other, PauliSentence):
            if other.pauli_rep is None:
                raise NotImplementedError(f'Cannot compute a native commutator of a Pauli word or sentence with the operator {other} of type {type(other)}.You can try to use qml.commutator(op1, op2, pauli=False) instead.')
            other = qml.pauli.pauli_sentence(other)
        for pw1 in self:
            for pw2 in other:
                comm_pw, coeff = pw1._commutator(pw2)
                if len(comm_pw) != 0:
                    final_ps[comm_pw] += coeff * self[pw1] * other[pw2]
        return final_ps

    def __str__(self):
        """String representation of the PauliSentence."""
        if len(self) == 0:
            return '0 * I'
        return '\n+ '.join((f'{coeff} * {str(pw)}' for pw, coeff in self.items()))

    def __repr__(self):
        """Terminal representation for PauliSentence"""
        return str(self)

    @property
    def wires(self):
        """Track wires of the PauliSentence."""
        return Wires.all_wires((pw.wires for pw in self.keys()))

    def to_mat(self, wire_order=None, format='dense', buffer_size=None):
        """Returns the matrix representation.

        Keyword Args:
            wire_order (iterable or None): The order of qubits in the tensor product.
            format (str): The format of the matrix. It is "dense" by default. Use "csr" for sparse.
            buffer_size (int or None): The maximum allowed memory in bytes to store intermediate results
                in the calculation of sparse matrices. It defaults to ``2 ** 30`` bytes that make
                1GB of memory. In general, larger buffers allow faster computations.

        Returns:
            (Union[NumpyArray, ScipySparseArray]): Matrix representation of the Pauli sentence.

        Raises:
            ValueError: Can't get the matrix of an empty PauliSentence.
        """
        wire_order = self.wires if wire_order is None else Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise ValueError(f"Can't get the matrix for the specified wire order because it does not contain all the Pauli sentence's wires {self.wires}")

        def _pw_wires(w: Iterable) -> Wires:
            """Return the native Wires instance for a list of wire labels.
            w represents the wires of the PauliWord being processed. In case
            the PauliWord is empty ({}), choose any arbitrary wire from the
            PauliSentence it is composed in.
            """
            return w or Wires(self.wires[0]) if self.wires else self.wires
        if len(self) == 0:
            n = len(wire_order) if wire_order is not None else 0
            if format == 'dense':
                return np.zeros((2 ** n, 2 ** n))
            return sparse.csr_matrix((2 ** n, 2 ** n), dtype='complex128')
        if format != 'dense':
            return self._to_sparse_mat(wire_order, buffer_size=buffer_size)
        mats_and_wires_gen = ((coeff * pw.to_mat(wire_order=_pw_wires(pw.wires), format=format), _pw_wires(pw.wires)) for pw, coeff in self.items())
        reduced_mat, result_wire_order = math.reduce_matrices(mats_and_wires_gen=mats_and_wires_gen, reduce_func=math.add)
        return math.expand_matrix(reduced_mat, result_wire_order, wire_order=wire_order)

    def _to_sparse_mat(self, wire_order, buffer_size=None):
        """Compute the sparse matrix of the Pauli sentence by efficiently adding the Pauli words
        that it is composed of. See pauli_sparse_matrices.md for the technical details."""
        pauli_words = list(self)
        n_wires = len(wire_order)
        matrix_size = 2 ** n_wires
        matrix = sparse.csr_matrix((matrix_size, matrix_size), dtype='complex128')
        op_sparse_idx = _ps_to_sparse_index(pauli_words, wire_order)
        _, unique_sparse_structures, unique_invs = np.unique(op_sparse_idx, axis=0, return_index=True, return_inverse=True)
        pw_sparse_structures = unique_sparse_structures[unique_invs]
        buffer_size = buffer_size or 2 ** 30
        buffer_size = max(1, buffer_size // ((16 + 8) * matrix_size))
        mat_data = np.empty((matrix_size, buffer_size), dtype=np.complex128)
        mat_indices = np.empty((matrix_size, buffer_size), dtype=np.int64)
        n_matrices_in_buffer = 0
        for sparse_structure in unique_sparse_structures:
            indices, *_ = np.nonzero(pw_sparse_structures == sparse_structure)
            mat = self._sum_same_structure_pws([pauli_words[i] for i in indices], wire_order)
            mat_data[:, n_matrices_in_buffer] = mat.data
            mat_indices[:, n_matrices_in_buffer] = mat.indices
            n_matrices_in_buffer += 1
            if n_matrices_in_buffer == buffer_size:
                matrix += self._sum_different_structure_pws(mat_indices, mat_data)
                n_matrices_in_buffer = 0
        matrix += self._sum_different_structure_pws(mat_indices[:, :n_matrices_in_buffer], mat_data[:, :n_matrices_in_buffer])
        matrix.eliminate_zeros()
        return matrix

    def dot(self, vector, wire_order=None):
        """Computes the matrix-vector product of the Pauli sentence with a state vector.
        See pauli_sparse_matrices.md for the technical details."""
        wire_order = self.wires if wire_order is None else Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise ValueError(f"Can't get the matrix for the specified wire order because it does not contain all the Pauli sentence's wires {self.wires}")
        pauli_words = list(self)
        op_sparse_idx = _ps_to_sparse_index(pauli_words, wire_order)
        _, unique_sparse_structures, unique_invs = np.unique(op_sparse_idx, axis=0, return_index=True, return_inverse=True)
        pw_sparse_structures = unique_sparse_structures[unique_invs]
        dtype = np.complex64 if vector.dtype in (np.float32, np.complex64) else np.complex128
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        mv = np.zeros_like(vector, dtype=dtype)
        for sparse_structure in unique_sparse_structures:
            indices, *_ = np.nonzero(pw_sparse_structures == sparse_structure)
            entries, data = self._get_same_structure_csr([pauli_words[i] for i in indices], wire_order)
            mv += vector[:, entries] * data.reshape(1, -1)
        return mv.reshape(vector.shape)

    def _get_same_structure_csr(self, pauli_words, wire_order):
        """Returns the CSR indices and data for Pauli words with the same sparse structure."""
        indices = pauli_words[0]._get_csr_indices(wire_order)
        nwires = len(wire_order)
        nwords = len(pauli_words)
        inner = np.empty((nwords, 2 ** (nwires - nwires // 2)), dtype=np.complex128)
        outer = np.empty((nwords, 2 ** (nwires // 2)), dtype=np.complex128)
        for i, word in enumerate(pauli_words):
            outer[i, :], inner[i, :] = word._get_csr_data_2(wire_order, coeff=qml.math.to_numpy(self[word]))
        data = outer.T @ inner
        return (indices, data.ravel())

    def _sum_same_structure_pws(self, pauli_words, wire_order):
        """Sums Pauli words with the same sparse structure."""
        mat = pauli_words[0].to_mat(wire_order, coeff=qml.math.to_numpy(self[pauli_words[0]]), format='csr')
        for word in pauli_words[1:]:
            mat.data += word.to_mat(wire_order, coeff=qml.math.to_numpy(self[word]), format='csr').data
        return mat

    @staticmethod
    def _sum_different_structure_pws(indices, data):
        """Sums Pauli words with different parse structures."""
        size = indices.shape[0]
        idx = np.argsort(indices, axis=1)
        matrix = sparse.csr_matrix((size, size), dtype='complex128')
        matrix.indices = np.take_along_axis(indices, idx, axis=1).ravel()
        matrix.data = np.take_along_axis(data, idx, axis=1).ravel()
        num_entries_per_row = indices.shape[1]
        matrix.indptr = _cached_arange(size + 1) * num_entries_per_row
        matrix.data[np.abs(matrix.data) < 1e-16] = 0
        matrix.eliminate_zeros()
        return matrix

    def operation(self, wire_order=None):
        """Returns a native PennyLane :class:`~pennylane.operation.Operation` representing the PauliSentence."""
        if len(self) == 0:
            return qml.s_prod(0, Identity(wires=wire_order))
        summands = []
        wire_order = wire_order or self.wires
        for pw, coeff in self.items():
            pw_op = pw.operation(wire_order=list(wire_order))
            rep = PauliSentence({pw: coeff})
            summands.append(pw_op if coeff == 1 else SProd(coeff, pw_op, _pauli_rep=rep))
        return summands[0] if len(summands) == 1 else Sum(*summands, _pauli_rep=self)

    def hamiltonian(self, wire_order=None):
        """Returns a native PennyLane :class:`~pennylane.Hamiltonian` representing the PauliSentence."""
        if len(self) == 0:
            if wire_order in (None, [], Wires([])):
                raise ValueError("Can't get the Hamiltonian for an empty PauliSentence.")
            return Hamiltonian([], [])
        wire_order = wire_order or self.wires
        wire_order = list(wire_order)
        return Hamiltonian(list(self.values()), [pw.operation(wire_order=wire_order, get_as_tensor=True) for pw in self])

    def simplify(self, tol=1e-08):
        """Remove any PauliWords in the PauliSentence with coefficients less than the threshold tolerance."""
        items = list(self.items())
        for pw, coeff in items:
            if abs(coeff) <= tol:
                del self[pw]

    def map_wires(self, wire_map: dict) -> 'PauliSentence':
        """Return a new PauliSentence with the wires mapped."""
        return self.__class__({pw.map_wires(wire_map): coeff for pw, coeff in self.items()})