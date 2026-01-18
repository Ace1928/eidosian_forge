from collections import defaultdict
from typing import (
import numbers
import numpy as np
from sympy.logic.boolalg import And, Not, Or, Xor
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from scipy.sparse import csr_matrix
from cirq import linalg, protocols, qis, value
from cirq._doc import document
from cirq.linalg import operator_spaces
from cirq.ops import identity, raw_types, pauli_gates, pauli_string
from cirq.ops.pauli_string import PauliString, _validate_qubit_mapping
from cirq.ops.projector import ProjectorString
from cirq.value.linear_dict import _format_terms
@value.value_equality(approximate=True, unhashable=True)
class PauliSum:
    """Represents operator defined by linear combination of PauliStrings.

    Since `cirq.PauliString`s store their own coefficients, this class
    does not implement the `cirq.LinearDict` interface. Instead, you can
    add and subtract terms and then iterate over the resulting
    (simplified) expression.

    Under the hood, this class is backed by a LinearDict with coefficient-less
    PauliStrings as keys. PauliStrings are reconstructed on-the-fly during
    iteration.

    PauliSums can be constructed explicitly:


    >>> a, b = cirq.GridQubit.rect(1, 2)
    >>> psum = cirq.PauliSum.from_pauli_strings([
    ...     cirq.PauliString(-1, cirq.X(a), cirq.Y(b)),
    ...     cirq.PauliString(2, cirq.Z(a), cirq.Z(b)),
    ...     cirq.PauliString(0.5, cirq.Y(a), cirq.Y(b))
    ... ])
    >>> print(psum)
    -1.000*X(q(0, 0))*Y(q(0, 1))+2.000*Z(q(0, 0))*Z(q(0, 1))+0.500*Y(q(0, 0))*Y(q(0, 1))


    or implicitly:


    >>> a, b = cirq.GridQubit.rect(1, 2)
    >>> psum = cirq.X(a) * cirq.X(b) + 3.0 * cirq.Y(a)
    >>> print(psum)
    1.000*X(q(0, 0))*X(q(0, 1))+3.000*Y(q(0, 0))

    basic arithmetic and expectation operations are supported as well:


    >>> a, b = cirq.GridQubit.rect(1, 2)
    >>> psum = cirq.X(a) * cirq.X(b) + 3.0 * cirq.Y(a)
    >>> two_psum = 2 * psum
    >>> four_psum = two_psum + two_psum
    >>> print(four_psum)
    4.000*X(q(0, 0))*X(q(0, 1))+12.000*Y(q(0, 0))


    >>> expectation = four_psum.expectation_from_state_vector(
    ...     np.array([0.707106, 0, 0, 0.707106], dtype=complex),
    ...     qubit_map={a: 0, b: 1}
    ... )
    >>> print(f'{expectation:.1f}')
    4.0+0.0j
    """

    def __init__(self, linear_dict: Optional[value.LinearDict[UnitPauliStringT]]=None):
        """Construct a PauliSum from a linear dictionary.

        Note, the preferred method of constructing PauliSum objects is either implicitly
        or via the `from_pauli_strings` function.

        Args:
            linear_dict: Set of  (`cirq.Qid`, `cirq.Pauli`) tuples to construct the sum
                from.

        Raises:
            ValueError: If structure of `linear_dict` contains tuples other than the
                form (`cirq.Qid`, `cirq.Pauli`).
        """
        if linear_dict is None:
            linear_dict = value.LinearDict()
        if not _is_linear_dict_of_unit_pauli_string(linear_dict):
            raise ValueError('PauliSum constructor takes a LinearDict[UnitPauliStringT]. Consider using PauliSum.from_pauli_strings() or adding and subtracting PauliStrings')
        self._linear_dict = linear_dict

    def _value_equality_values_(self):
        return self._linear_dict

    @staticmethod
    def wrap(val: PauliSumLike) -> 'PauliSum':
        """Convert a `cirq.PauliSumLike` object to a PauliSum

        Attempts to convert an existing int, float, complex, `cirq.PauliString`,
        `cirq.PauliSum` or `cirq.SingleQubitPauliStringGateOperation` into
        a `cirq.PauliSum` object. For example:


        >>> my_psum = cirq.PauliSum.wrap(2.345)
        >>> my_psum
        cirq.PauliSum(cirq.LinearDict({frozenset(): (2.345+0j)}))


        Args:
            `cirq.PauliSumLike` to convert to PauliSum.

        Returns:
            PauliSum representation of `val`.
        """
        if isinstance(val, PauliSum):
            return val
        return PauliSum() + val

    @classmethod
    def from_pauli_strings(cls, terms: Union[PauliString, List[PauliString]]) -> 'PauliSum':
        """Returns a PauliSum by combining `cirq.PauliString` terms.

        Args:
            terms: `cirq.PauliString` or List of `cirq.PauliString`s to use inside
                of this PauliSum object.
        Returns:
            PauliSum object representing the addition of all the `cirq.PauliString`
                terms in `terms`.
        """
        if isinstance(terms, PauliString):
            terms = [terms]
        termdict: DefaultDict[UnitPauliStringT, value.Scalar] = defaultdict(lambda: 0)
        for pstring in terms:
            key = frozenset(pstring._qubit_pauli_map.items())
            termdict[key] += pstring.coefficient
        return cls(linear_dict=value.LinearDict(termdict))

    @classmethod
    def from_boolean_expression(cls, boolean_expr: Expr, qubit_map: Dict[str, 'cirq.Qid']) -> 'PauliSum':
        """Builds the Hamiltonian representation of a Boolean expression.

        This is based on "On the representation of Boolean and real functions as Hamiltonians for
        quantum computing" by Stuart Hadfield, https://arxiv.org/abs/1804.09130

        Args:
            boolean_expr: A Sympy expression containing symbols and Boolean operations
            qubit_map: map of string (boolean variable name) to qubit.

        Return:
            The PauliSum that represents the Boolean expression.

        Raises:
            ValueError: If `boolean_expr` is of an unsupported type.
        """
        if isinstance(boolean_expr, Symbol):
            return cls.from_pauli_strings([PauliString({}, 0.5), PauliString({qubit_map[boolean_expr.name]: pauli_gates.Z}, -0.5)])
        if isinstance(boolean_expr, (And, Not, Or, Xor)):
            sub_pauli_sums = [cls.from_boolean_expression(sub_boolean_expr, qubit_map) for sub_boolean_expr in boolean_expr.args]
            if isinstance(boolean_expr, And):
                pauli_sum = cls.from_pauli_strings(PauliString({}, 1.0))
                for sub_pauli_sum in sub_pauli_sums:
                    pauli_sum = pauli_sum * sub_pauli_sum
            elif isinstance(boolean_expr, Not):
                assert len(sub_pauli_sums) == 1
                pauli_sum = cls.from_pauli_strings(PauliString({}, 1.0)) - sub_pauli_sums[0]
            elif isinstance(boolean_expr, Or):
                pauli_sum = cls.from_pauli_strings(PauliString({}, 0.0))
                for sub_pauli_sum in sub_pauli_sums:
                    pauli_sum = pauli_sum + sub_pauli_sum - pauli_sum * sub_pauli_sum
            elif isinstance(boolean_expr, Xor):
                pauli_sum = cls.from_pauli_strings(PauliString({}, 0.0))
                for sub_pauli_sum in sub_pauli_sums:
                    pauli_sum = pauli_sum + sub_pauli_sum - 2.0 * pauli_sum * sub_pauli_sum
            return pauli_sum
        raise ValueError(f'Unsupported type: {type(boolean_expr)}')

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        """The sorted list of qubits used in this PauliSum."""
        qs = {q for k in self._linear_dict.keys() for q, _ in k}
        return tuple(sorted(qs))

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'PauliSum':
        """Return a new PauliSum on `new_qubits`.

        Args:
            *new_qubits: `cirq.Qid` objects to replace existing
                qubit objects in this PauliSum.

        Returns:
            PauliSum with new_qubits replacing the previous
                qubits.

        Raises:
            ValueError: If len(new_qubits) != len(self.qubits).

        """
        qubits = self.qubits
        if len(new_qubits) != len(qubits):
            raise ValueError('Incorrect number of qubits for PauliSum.')
        qubit_map = dict(zip(qubits, new_qubits))
        new_pauli_strings = []
        for pauli_string in self:
            new_pauli_strings.append(pauli_string.map_qubits(qubit_map))
        return PauliSum.from_pauli_strings(new_pauli_strings)

    def copy(self) -> 'PauliSum':
        """Return a copy of this PauliSum.

        Returns: A copy of this PauliSum.
        """
        factory = type(self)
        return factory(self._linear_dict.copy())

    def matrix(self, qubits: Optional[Iterable[raw_types.Qid]]=None) -> np.ndarray:
        """Returns the matrix of this PauliSum in computational basis of qubits.

        Args:
            qubits: Ordered collection of qubits that determine the subspace
                in which the matrix representation of the Pauli sum is to
                be computed. If none is provided the default ordering of
                `self.qubits` is used.  Qubits present in `qubits` but absent from
                `self.qubits` are acted on by the identity.

        Returns:
            np.ndarray representing the matrix of this PauliSum expression.

        Raises:
            TypeError: if any of the gates in self does not provide a unitary.
        """
        qubits = self.qubits if qubits is None else tuple(qubits)
        num_qubits = len(qubits)
        num_dim = 2 ** num_qubits
        result = np.zeros((num_dim, num_dim), dtype=np.complex128)
        for vec, coeff in self._linear_dict.items():
            op = _pauli_string_from_unit(vec)
            result += coeff * op.matrix(qubits)
        return result

    def _has_unitary_(self) -> bool:
        return linalg.is_unitary(self.matrix())

    def _unitary_(self) -> np.ndarray:
        m = self.matrix()
        if linalg.is_unitary(m):
            return m
        raise ValueError(f'{self} is not unitary')

    def _json_dict_(self):

        def key_json(k: UnitPauliStringT):
            return [list(e) for e in sorted(k)]
        return {'items': list(((key_json(k), v) for k, v in self._linear_dict.items()))}

    @classmethod
    def _from_json_dict_(cls, items, **kwargs):
        mapping = {frozenset((tuple(qid_pauli) for qid_pauli in unit_pauli_string)): val for unit_pauli_string, val in items}
        return cls(linear_dict=value.LinearDict(mapping))

    def expectation_from_state_vector(self, state_vector: np.ndarray, qubit_map: Mapping[raw_types.Qid, int], *, atol: float=1e-07, check_preconditions: bool=True) -> float:
        """Evaluate the expectation of this PauliSum given a state vector.

        See `PauliString.expectation_from_state_vector`.

        Args:
            state_vector: An array representing a valid state vector.
            qubit_map: A map from all qubits used in this PauliSum to the
                indices of the qubits that `state_vector` is defined over.
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state_vector` represents
                a valid state vector.

        Returns:
            The expectation value of the input state.

        Raises:
            NotImplementedError: If any of the coefficients are imaginary,
                so that this is not Hermitian.
            TypeError: If the input state is not a complex type.
            ValueError: If the input vector is not the correct size or shape.
        """
        if any((abs(p.coefficient.imag) > 0.0001 for p in self)):
            raise NotImplementedError(f'Cannot compute expectation value of a non-Hermitian PauliString <{self}>. Coefficient must be real.')
        if state_vector.dtype.kind != 'c':
            raise TypeError('Input state dtype must be np.complex64 or np.complex128')
        size = state_vector.size
        num_qubits = size.bit_length() - 1
        _validate_qubit_mapping(qubit_map, self.qubits, num_qubits)
        if len(state_vector.shape) != 1 and state_vector.shape != (2,) * num_qubits:
            raise ValueError('Input array does not represent a state vector with shape `(2 ** n,)` or `(2, ..., 2)`.')
        if check_preconditions:
            qis.validate_normalized_state_vector(state_vector=state_vector, qid_shape=(2,) * num_qubits, dtype=state_vector.dtype, atol=atol)
        return sum((p._expectation_from_state_vector_no_validation(state_vector, qubit_map) for p in self))

    def expectation_from_density_matrix(self, state: np.ndarray, qubit_map: Mapping[raw_types.Qid, int], *, atol: float=1e-07, check_preconditions: bool=True) -> float:
        """Evaluate the expectation of this PauliSum given a density matrix.

        See `PauliString.expectation_from_density_matrix`.

        Args:
            state: An array representing a valid  density matrix.
            qubit_map: A map from all qubits used in this PauliSum to the
                indices of the qubits that `state` is defined over.
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state` represents a
                valid density matrix.

        Returns:
            The expectation value of the input state.

        Raises:
            NotImplementedError: If any of the coefficients are imaginary,
                so that this is not Hermitian.
            TypeError: If the input state is not a complex type.
            ValueError: If the input vector is not the correct size or shape.
        """
        if any((abs(p.coefficient.imag) > 0.0001 for p in self)):
            raise NotImplementedError(f'Cannot compute expectation value of a non-Hermitian PauliString <{self}>. Coefficient must be real.')
        if state.dtype.kind != 'c':
            raise TypeError('Input state dtype must be np.complex64 or np.complex128')
        size = state.size
        num_qubits = int(np.sqrt(size)).bit_length() - 1
        _validate_qubit_mapping(qubit_map, self.qubits, num_qubits)
        dim = int(np.sqrt(size))
        if state.shape != (dim, dim) and state.shape != (2, 2) * num_qubits:
            raise ValueError('Input array does not represent a density matrix with shape `(2 ** n, 2 ** n)` or `(2, ..., 2)`.')
        if check_preconditions:
            _ = qis.to_valid_density_matrix(density_matrix_rep=state.reshape(dim, dim), num_qubits=num_qubits, dtype=state.dtype, atol=atol)
        return sum((p._expectation_from_density_matrix_no_validation(state, qubit_map) for p in self))

    def __iter__(self):
        for vec, coeff in self._linear_dict.items():
            yield _pauli_string_from_unit(vec, coeff)

    def __len__(self) -> int:
        return len(self._linear_dict)

    def __iadd__(self, other):
        if isinstance(other, numbers.Complex):
            other = PauliSum.from_pauli_strings([PauliString(coefficient=other)])
        elif isinstance(other, PauliString):
            other = PauliSum.from_pauli_strings([other])
        if not isinstance(other, PauliSum):
            return NotImplemented
        self._linear_dict += other._linear_dict
        return self

    def __add__(self, other):
        if not isinstance(other, (numbers.Complex, PauliString, PauliSum)):
            if hasattr(other, 'gate') and isinstance(other.gate, identity.IdentityGate):
                other = PauliString(other)
            else:
                return NotImplemented
        result = self.copy()
        result += other
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __isub__(self, other):
        if isinstance(other, numbers.Complex):
            other = PauliSum.from_pauli_strings([PauliString(coefficient=other)])
        if isinstance(other, PauliString):
            other = PauliSum.from_pauli_strings([other])
        if not isinstance(other, PauliSum):
            return NotImplemented
        self._linear_dict -= other._linear_dict
        return self

    def __sub__(self, other):
        if not isinstance(other, (numbers.Complex, PauliString, PauliSum)):
            return NotImplemented
        result = self.copy()
        result -= other
        return result

    def __neg__(self):
        factory = type(self)
        return factory(-self._linear_dict)

    def __imul__(self, other: PauliSumLike):
        if not isinstance(other, (numbers.Complex, PauliString, PauliSum)):
            return NotImplemented
        if isinstance(other, numbers.Complex):
            self._linear_dict *= other
        elif isinstance(other, PauliString):
            temp = PauliSum.from_pauli_strings([term * other for term in self])
            self._linear_dict = temp._linear_dict
        elif isinstance(other, PauliSum):
            temp = PauliSum.from_pauli_strings([term * other_term for term in self for other_term in other])
            self._linear_dict = temp._linear_dict
        return self

    def __mul__(self, other: PauliSumLike):
        if not isinstance(other, (numbers.Complex, PauliString, PauliSum)):
            return NotImplemented
        result = self.copy()
        result *= other
        return result

    def __rmul__(self, other: PauliSumLike):
        if isinstance(other, numbers.Complex):
            result = self.copy()
            result *= other
            return result
        elif isinstance(other, PauliString):
            result = self.copy()
            return PauliSum.from_pauli_strings([other]) * result
        return NotImplemented

    def __pow__(self, exponent: int):
        if not isinstance(exponent, numbers.Integral):
            return NotImplemented
        if exponent == 0:
            return PauliSum(value.LinearDict({frozenset(): 1 + 0j}))
        if exponent > 0:
            result = self.copy()
            for _ in range(exponent - 1):
                result *= self
            return result
        return NotImplemented

    def __truediv__(self, a: value.Scalar):
        return self.__mul__(1 / a)

    def __bool__(self) -> bool:
        return bool(self._linear_dict)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'cirq.{class_name}({self._linear_dict!r})'

    def __format__(self, format_spec: str) -> str:
        terms = [(_pauli_string_from_unit(v), self._linear_dict[v]) for v in self._linear_dict.keys()]
        return _format_terms(terms=terms, format_spec=format_spec)

    def __str__(self) -> str:
        return self.__format__('.3f')