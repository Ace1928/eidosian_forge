import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
@value.value_equality(approximate=True, manual_cls=True)
class PauliString(raw_types.Operation, Generic[TKey]):
    """Represents a multi-qubit pauli operator or pauli observable.

    `cirq.PauliString` represents a multi-qubit pauli operator, i.e.
    a tensor product of single qubit (non identity) pauli operations,
    each acting on a different qubit. For  example,

    - X(0) * Y(1) * Z(2): Represents a pauli string which is a tensor product of
                          `cirq.X(q0)`, `cirq.Y(q1)` and `cirq.Z(q2)`.

    If more than one pauli operation acts on the same set of qubits, their composition is
    immediately reduced to an equivalent (possibly multi-qubit) Pauli operator. Also, identity
    operations are dropped by the `PauliString` class. For example:

    >>> a, b = cirq.LineQubit.range(2)
    >>> print(cirq.X(a) * cirq.Y(b)) # Tensor product of Pauli's acting on different qubits.
    X(q(0))*Y(q(1))
    >>> print(cirq.X(a) * cirq.Y(a)) # Composition is reduced to an equivalent PauliString.
    1j*Z(q(0))
    >>> print(cirq.X(a) * cirq.I(b)) # Identity operations are dropped by default.
    X(q(0))
    >>> print(cirq.PauliString()) # String representation of an "empty" PaulString is "I".
    I

    `cirq.PauliString` is often used to represent:
    - Pauli operators: Can be inserted into circuits as multi qubit operations.
    - Pauli observables: Can be measured using either `cirq.measure_single_paulistring`/
                        `cirq.measure_paulistring_terms`; or using the observable
                        measurement framework in `cirq.measure_observables`.

    PauliStrings can be constructed via various different ways, some examples are
    given as follows:

    >>> a, b, c = cirq.LineQubit.range(3)
    >>> print(cirq.PauliString([cirq.X(a), cirq.X(a)]))
    I
    >>> print(cirq.PauliString(-1, cirq.X(a), cirq.Y(b), cirq.Z(c)))
    -X(q(0))*Y(q(1))*Z(q(2))
    >>> print(-1 * cirq.X(a) * cirq.Y(b) * cirq.Z(c))
    -X(q(0))*Y(q(1))*Z(q(2))
    >>> print(cirq.PauliString({a: cirq.X}, [-2, 3, cirq.Y(a)]))
    -6j*Z(q(0))
    >>> print(cirq.PauliString({a: cirq.I, b: cirq.X}))
    X(q(1))
    >>> print(cirq.PauliString({a: cirq.Y}, qubit_pauli_map={a: cirq.X}))
    1j*Z(q(0))

    Note that `cirq.PauliString`s are immutable objects. If you need a mutable version
    of pauli strings, see `cirq.MutablePauliString`.
    """

    def __init__(self, *contents: 'cirq.PAULI_STRING_LIKE', qubit_pauli_map: Optional[Dict[TKey, 'cirq.Pauli']]=None, coefficient: 'cirq.TParamValComplex'=1):
        """Initializes a new `PauliString` operation.

        Args:
            *contents: A value or values to convert into a pauli string. This
                can be a number, a pauli operation, a dictionary from qubit to
                pauli/identity gates, or collections thereof. If a list of
                values is given, they are each individually converted and then
                multiplied from left to right in order.
            qubit_pauli_map: Initial dictionary mapping qubits to pauli
                operations. Defaults to the empty dictionary. Note that, unlike
                dictionaries passed to contents, this dictionary must not
                contain any identity gate values. Further note that this
                argument specifies values that are logically *before* factors
                specified in `contents`; `contents` are *right* multiplied onto
                the values in this dictionary.
            coefficient: Initial scalar coefficient or symbol. Defaults to 1.

        Raises:
            TypeError: If the `qubit_pauli_map` has values that are not Paulis.
        """
        if qubit_pauli_map is not None:
            for v in qubit_pauli_map.values():
                if not isinstance(v, pauli_gates.Pauli):
                    raise TypeError(f'{v} is not a Pauli')
        self._qubit_pauli_map: Dict[TKey, 'cirq.Pauli'] = qubit_pauli_map or {}
        self._coefficient: Union['cirq.TParamValComplex', sympy.Expr] = coefficient if isinstance(coefficient, sympy.Expr) else complex(coefficient)
        if contents:
            m = self.mutable_copy().inplace_left_multiply_by(contents).frozen()
            self._qubit_pauli_map = m._qubit_pauli_map
            self._coefficient = m._coefficient

    @property
    def coefficient(self) -> 'cirq.TParamValComplex':
        """A scalar coefficient or symbol."""
        return self._coefficient

    def _value_equality_values_(self):
        if len(self._qubit_pauli_map) == 1 and self.coefficient == 1:
            q, p = list(self._qubit_pauli_map.items())[0]
            return gate_operation.GateOperation(p, [q])._value_equality_values_()
        return (frozenset(self._qubit_pauli_map.items()), self._coefficient)

    def _json_dict_(self) -> Dict[str, Any]:
        return {'qubit_pauli_map': list(self._qubit_pauli_map.items()), 'coefficient': self.coefficient}

    @classmethod
    def _from_json_dict_(cls, qubit_pauli_map, coefficient, **kwargs):
        return cls(qubit_pauli_map=dict(qubit_pauli_map), coefficient=coefficient)

    def _value_equality_values_cls_(self):
        if len(self._qubit_pauli_map) == 1 and self.coefficient == 1:
            return gate_operation.GateOperation
        return PauliString

    def equal_up_to_coefficient(self, other: 'cirq.PauliString') -> bool:
        """Returns true of `self` and `other` are equal pauli strings, ignoring the coefficient."""
        return self._qubit_pauli_map == other._qubit_pauli_map

    def __getitem__(self, key: TKey) -> pauli_gates.Pauli:
        return self._qubit_pauli_map[key]

    @overload
    def get(self, key: Any, default: None=None) -> Optional[pauli_gates.Pauli]:
        pass

    @overload
    def get(self, key: Any, default: TDefault) -> Union[pauli_gates.Pauli, TDefault]:
        pass

    def get(self, key: Any, default: Optional[TDefault]=None) -> Union[pauli_gates.Pauli, TDefault, None]:
        """Returns the `cirq.Pauli` operation acting on qubit `key` or `default` if none exists."""
        return self._qubit_pauli_map.get(key, default)

    @overload
    def __mul__(self, other: 'cirq.PauliString[TKeyOther]') -> 'cirq.PauliString[Union[TKey, TKeyOther]]':
        pass

    @overload
    def __mul__(self, other: Mapping[TKeyOther, 'cirq.PAULI_GATE_LIKE']) -> 'cirq.PauliString[Union[TKey, TKeyOther]]':
        pass

    @overload
    def __mul__(self, other: Iterable['cirq.PAULI_STRING_LIKE']) -> 'cirq.PauliString[Union[TKey, cirq.Qid]]':
        pass

    @overload
    def __mul__(self, other: 'cirq.Operation') -> 'cirq.PauliString[Union[TKey, cirq.Qid]]':
        pass

    @overload
    def __mul__(self, other: Union[complex, int, float, numbers.Number]) -> 'cirq.PauliString[TKey]':
        pass

    def __mul__(self, other):
        known = False
        if isinstance(other, raw_types.Operation) and isinstance(other.gate, identity.IdentityGate):
            known = True
        elif isinstance(other, (PauliString, numbers.Number)):
            known = True
        if known:
            return PauliString(cast(PAULI_STRING_LIKE, other), qubit_pauli_map=self._qubit_pauli_map, coefficient=self.coefficient)
        return NotImplemented

    @property
    def gate(self) -> 'cirq.DensePauliString':
        """Returns a `cirq.DensePauliString`"""
        order: List[Optional[pauli_gates.Pauli]] = [None, pauli_gates.X, pauli_gates.Y, pauli_gates.Z]
        from cirq.ops.dense_pauli_string import DensePauliString
        return DensePauliString(coefficient=self.coefficient, pauli_mask=[order.index(self[q]) for q in self.qubits])

    def __rmul__(self, other) -> 'PauliString':
        if isinstance(other, numbers.Number):
            return PauliString(qubit_pauli_map=self._qubit_pauli_map, coefficient=self._coefficient * complex(cast(SupportsComplex, other)))
        if isinstance(other, raw_types.Operation) and isinstance(other.gate, identity.IdentityGate):
            return self
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return PauliString(qubit_pauli_map=self._qubit_pauli_map, coefficient=self._coefficient / complex(cast(SupportsComplex, other)))
        return NotImplemented

    def __add__(self, other):
        return linear_combinations.PauliSum.from_pauli_strings(self).__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return linear_combinations.PauliSum.from_pauli_strings(self).__sub__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __contains__(self, key: TKey) -> bool:
        return key in self._qubit_pauli_map

    def _decompose_(self):
        if not self._has_unitary_():
            return None
        return [*([] if self.coefficient == 1 else [global_phase_op.global_phase_operation(self.coefficient)]), *[self[q].on(q) for q in self.qubits]]

    def keys(self) -> KeysView[TKey]:
        """Returns the sequence of qubits on which this pauli string acts."""
        return self._qubit_pauli_map.keys()

    @property
    def qubits(self) -> Tuple[TKey, ...]:
        """Returns a tuple of qubits on which this pauli string acts."""
        return tuple(self.keys())

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> List[str]:
        if not len(self._qubit_pauli_map):
            return NotImplemented
        qs = args.known_qubits or list(self.keys())
        symbols = list((str(self.get(q)) for q in qs))
        if self.coefficient == 1:
            prefix = '+'
        elif self.coefficient == -1:
            prefix = '-'
        elif self.coefficient == 1j:
            prefix = 'i'
        elif self.coefficient == -1j:
            prefix = '-i'
        elif isinstance(self.coefficient, numbers.Number):
            prefix = f'({args.format_complex(self.coefficient)})*'
        else:
            prefix = f'({self.coefficient})*'
        symbols[0] = f'PauliString({prefix}{symbols[0]})'
        return symbols

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'PauliString':
        """Returns a new `PauliString` with `self.qubits` mapped to `new_qubits`.

        Args:
            new_qubits: The new qubits to replace `self.qubits` by.

        Returns:
            `PauliString` with mapped qubits.

        Raises:
            ValueError: If `len(new_qubits) != len(self.qubits)`.
        """
        if len(new_qubits) != len(self.qubits):
            raise ValueError(f'Number of new qubits: {len(new_qubits)} does not match self.qubits: {len(self.qubits)}.')
        return PauliString(qubit_pauli_map=dict(zip(new_qubits, (self[q] for q in self.qubits))), coefficient=self._coefficient)

    def with_coefficient(self, new_coefficient: 'cirq.TParamValComplex') -> 'PauliString':
        """Returns a new `PauliString` with `self.coefficient` replaced with `new_coefficient`."""
        return PauliString(qubit_pauli_map=dict(self._qubit_pauli_map), coefficient=new_coefficient)

    def values(self) -> ValuesView[pauli_gates.Pauli]:
        """Ordered sequence of `cirq.Pauli` gates acting on `self.keys()`."""
        return self._qubit_pauli_map.values()

    def items(self) -> ItemsView[TKey, pauli_gates.Pauli]:
        """Returns (cirq.Qid, cirq.Pauli) pairs representing 1-qubit operations of pauli string."""
        return self._qubit_pauli_map.items()

    def frozen(self) -> 'cirq.PauliString':
        """Returns a `cirq.PauliString` with the same contents."""
        return self

    def mutable_copy(self) -> 'cirq.MutablePauliString':
        """Returns a new `cirq.MutablePauliString` with the same contents."""
        return MutablePauliString(coefficient=self.coefficient, pauli_int_dict={q: PAULI_GATE_LIKE_TO_INDEX_MAP[p] for q, p in self._qubit_pauli_map.items()})

    def __iter__(self) -> Iterator[TKey]:
        return iter(self._qubit_pauli_map.keys())

    def __bool__(self):
        return bool(self._qubit_pauli_map)

    def __len__(self) -> int:
        return len(self._qubit_pauli_map)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Print ASCII diagram in Jupyter."""
        if cycle:
            p.text('cirq.PauliString(...)')
        else:
            p.text(str(self))

    def __repr__(self) -> str:
        ordered_qubits = self.qubits
        prefix = ''
        factors = []
        if self._coefficient == -1:
            prefix = '-'
        else:
            factors.append(repr(self._coefficient))
        if not ordered_qubits:
            factors.append('cirq.PauliString()')
        for q in ordered_qubits:
            factors.append(repr(cast(raw_types.Gate, self[q]).on(q)))
        fused = prefix + '*'.join(factors)
        if len(factors) > 1:
            return f'({fused})'
        return fused

    def __str__(self) -> str:
        ordered_qubits = sorted(self.qubits)
        prefix = ''
        factors = []
        if self._coefficient == -1:
            prefix = '-'
        elif self._coefficient != 1:
            factors.append(repr(self._coefficient))
        if not ordered_qubits:
            factors.append('I')
        for q in ordered_qubits:
            factors.append(str(cast(raw_types.Gate, self[q]).on(q)))
        return prefix + '*'.join(factors)

    def matrix(self, qubits: Optional[Iterable[TKey]]=None) -> np.ndarray:
        """Returns the matrix of self in computational basis of qubits.

        Args:
            qubits: Ordered collection of qubits that determine the subspace
                in which the matrix representation of the Pauli string is to
                be computed. Qubits absent from `self.qubits` are acted on by
                the identity. Defaults to `self.qubits`.

        Raises:
            NotImplementedError: If this PauliString is parameterized.
        """
        qubits = self.qubits if qubits is None else qubits
        factors = [self.get(q, default=identity.I) for q in qubits]
        if cirq.is_parameterized(self):
            raise NotImplementedError('Cannot express as matrix when parameterized')
        assert isinstance(self.coefficient, complex)
        return linalg.kron(self.coefficient, *[protocols.unitary(f) for f in factors])

    def _has_unitary_(self) -> bool:
        if self._is_parameterized_():
            return False
        return abs(1 - abs(cast(complex, self.coefficient))) < 1e-06

    def _unitary_(self) -> Optional[np.ndarray]:
        if not self._has_unitary_():
            return None
        return self.matrix()

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs'):
        if not self._has_unitary_():
            return None
        assert isinstance(self.coefficient, complex)
        if self.coefficient != 1:
            args.target_tensor *= self.coefficient
        return protocols.apply_unitaries([self[q].on(q) for q in self.qubits], self.qubits, args)

    def expectation_from_state_vector(self, state_vector: np.ndarray, qubit_map: Mapping[TKey, int], *, atol: float=1e-07, check_preconditions: bool=True) -> float:
        """Evaluate the expectation of this PauliString given a state vector.

        Compute the expectation value of this PauliString with respect to a
        state vector. By convention expectation values are defined for Hermitian
        operators, and so this method will fail if this PauliString is
        non-Hermitian.

        `state` must be an array representation of a state vector and have
        shape `(2 ** n, )` or `(2, 2, ..., 2)` (n entries) where `state` is
        expressed over n qubits.

        `qubit_map` must assign an integer index to each qubit in this
        PauliString that determines which bit position of a computational basis
        state that qubit corresponds to. For example if `state` represents
        $|0\\rangle |+\\rangle$ and `q0, q1 = cirq.LineQubit.range(2)` then:

            cirq.X(q0).expectation(state, qubit_map={q0: 0, q1: 1}) = 0
            cirq.X(q0).expectation(state, qubit_map={q0: 1, q1: 0}) = 1

        Args:
            state_vector: An array representing a valid state vector.
            qubit_map: A map from all qubits used in this PauliString to the
                indices of the qubits that `state_vector` is defined over.
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state_vector` represents
                a valid state vector.

        Returns:
            The expectation value of the input state.

        Raises:
            NotImplementedError: If this PauliString is non-Hermitian or
                parameterized.
            TypeError: If the input state is not complex.
            ValueError: If the input state does not have the correct shape.
        """
        if self._is_parameterized_():
            raise NotImplementedError('Cannot get expectation value when parameterized')
        if abs(cast(complex, self.coefficient).imag) > 0.0001:
            raise NotImplementedError(f'Cannot compute expectation value of a non-Hermitian PauliString <{self}>. Coefficient must be real.')
        if state_vector.dtype.kind != 'c':
            raise TypeError('Input state dtype must be np.complex64 or np.complex128')
        size = state_vector.size
        num_qubits = size.bit_length() - 1
        if len(state_vector.shape) != 1 and state_vector.shape != (2,) * num_qubits:
            raise ValueError('Input array does not represent a state vector with shape `(2 ** n,)` or `(2, ..., 2)`.')
        _validate_qubit_mapping(qubit_map, self.qubits, num_qubits)
        if check_preconditions:
            qis.validate_normalized_state_vector(state_vector=state_vector, qid_shape=(2,) * num_qubits, dtype=state_vector.dtype, atol=atol)
        return self._expectation_from_state_vector_no_validation(state_vector, qubit_map)

    def _expectation_from_state_vector_no_validation(self, state_vector: np.ndarray, qubit_map: Mapping[TKey, int]) -> float:
        """Evaluate the expectation of this PauliString given a state vector.

        This method does not provide input validation. See
        `PauliString.expectation_from_state_vector` for function description.

        Args:
            state_vector: An array representing a valid state vector.
            qubit_map: A map from all qubits used in this PauliString to the
            indices of the qubits that `state` is defined over.

        Returns:
            The expectation value of the input state.
        """
        if len(state_vector.shape) == 1:
            num_qubits = state_vector.shape[0].bit_length() - 1
            state_vector = np.reshape(state_vector, (2,) * num_qubits)
        ket = np.copy(state_vector)
        for qubit, pauli in self.items():
            buffer = np.empty(ket.shape, dtype=state_vector.dtype)
            args = protocols.ApplyUnitaryArgs(target_tensor=ket, available_buffer=buffer, axes=(qubit_map[qubit],))
            ket = protocols.apply_unitary(pauli, args)
        return self.coefficient * np.tensordot(state_vector.conj(), ket, axes=len(ket.shape)).item()

    def expectation_from_density_matrix(self, state: np.ndarray, qubit_map: Mapping[TKey, int], *, atol: float=1e-07, check_preconditions: bool=True) -> float:
        """Evaluate the expectation of this PauliString given a density matrix.

        Compute the expectation value of this PauliString with respect to an
        array representing a density matrix. By convention expectation values
        are defined for Hermitian operators, and so this method will fail if
        this PauliString is non-Hermitian.

        `state` must be an array representation of a density matrix and have
        shape `(2 ** n, 2 ** n)` or `(2, 2, ..., 2)` (2*n entries), where
        `state` is expressed over n qubits.

        `qubit_map` must assign an integer index to each qubit in this
        PauliString that determines which bit position of a computational basis
        state that qubit corresponds to. For example if `state` represents
        $|0\\rangle |+\\rangle$ and `q0, q1 = cirq.LineQubit.range(2)` then:

            cirq.X(q0).expectation(state, qubit_map={q0: 0, q1: 1}) = 0
            cirq.X(q0).expectation(state, qubit_map={q0: 1, q1: 0}) = 1

        Args:
            state: An array representing a valid  density matrix.
            qubit_map: A map from all qubits used in this PauliString to the
                indices of the qubits that `state` is defined over.
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state` represents a
                valid density matrix.

        Returns:
            The expectation value of the input state.

        Raises:
            NotImplementedError: If this PauliString is non-Hermitian or
                parameterized.
            TypeError: If the input state is not complex.
            ValueError: If the input state does not have the correct shape.
        """
        if self._is_parameterized_():
            raise NotImplementedError('Cannot get expectation value when parameterized')
        if abs(cast(complex, self.coefficient).imag) > 0.0001:
            raise NotImplementedError(f'Cannot compute expectation value of a non-Hermitian PauliString <{self}>. Coefficient must be real.')
        if state.dtype.kind != 'c':
            raise TypeError('Input state dtype must be np.complex64 or np.complex128')
        size = state.size
        num_qubits = int(np.sqrt(size)).bit_length() - 1
        dim = 1 << num_qubits
        if state.shape != (dim, dim) and state.shape != (2, 2) * num_qubits:
            raise ValueError('Input array does not represent a density matrix with shape `(2 ** n, 2 ** n)` or `(2, ..., 2)`.')
        _validate_qubit_mapping(qubit_map, self.qubits, num_qubits)
        if check_preconditions:
            _ = qis.to_valid_density_matrix(density_matrix_rep=state.reshape(dim, dim), num_qubits=num_qubits, dtype=state.dtype, atol=atol)
        return self._expectation_from_density_matrix_no_validation(state, qubit_map)

    def _expectation_from_density_matrix_no_validation(self, state: np.ndarray, qubit_map: Mapping[TKey, int]) -> float:
        """Evaluate the expectation of this PauliString given a density matrix.

        This method does not provide input validation. See
        `PauliString.expectation_from_density_matrix` for function description.

        Args:
            state: An array representing a valid  density matrix.
            qubit_map: A map from all qubits used in this PauliString to the
            indices of the qubits that `state` is defined over.

        Returns:
            The expectation value of the input state.
        """
        result = np.copy(state)
        if len(state.shape) == 2:
            num_qubits = state.shape[0].bit_length() - 1
            result = np.reshape(result, (2,) * num_qubits * 2)
        for qubit, pauli in self.items():
            buffer = np.empty(result.shape, dtype=state.dtype)
            args = protocols.ApplyUnitaryArgs(target_tensor=result, available_buffer=buffer, axes=(qubit_map[qubit],))
            result = protocols.apply_unitary(pauli, args)
        while any(result.shape):
            result = np.trace(result, axis1=0, axis2=len(result.shape) // 2)
        return float(np.real(result * self.coefficient))

    def zip_items(self, other: 'cirq.PauliString[TKey]') -> Iterator[Tuple[TKey, Tuple[pauli_gates.Pauli, pauli_gates.Pauli]]]:
        """Combines pauli operations from pauli strings in a qubit-by-qubit fashion.

        For every qubit that has a `cirq.Pauli` operation acting on it in both `self` and `other`,
        the method yields a tuple corresponding to `(qubit, (pauli_in_self, pauli_in_other))`.

        Args:
            other: The other `cirq.PauliString` to zip pauli operations with.

        Returns:
            A sequence of `(qubit, (pauli_in_self, pauli_in_other))` tuples for every `qubit`
            that has a `cirq.Pauli` operation acting on it in both `self` and `other.
        """
        for qubit, pauli0 in self.items():
            if qubit in other:
                yield (qubit, (pauli0, other[qubit]))

    def zip_paulis(self, other: 'cirq.PauliString') -> Iterator[Tuple[pauli_gates.Pauli, pauli_gates.Pauli]]:
        """Combines pauli operations from pauli strings in a qubit-by-qubit fashion.

        For every qubit that has a `cirq.Pauli` operation acting on it in both `self` and `other`,
        the method yields a tuple corresponding to `(pauli_in_self, pauli_in_other)`.

        Args:
            other: The other `cirq.PauliString` to zip pauli operations with.

        Returns:
            A sequence of `(pauli_in_self, pauli_in_other)` tuples for every `qubit`
            that has a `cirq.Pauli` operation acting on it in both `self` and `other.
        """
        return (paulis for qubit, paulis in self.zip_items(other))

    def _commutes_(self, other: Any, *, atol: float=1e-08) -> Union[bool, NotImplementedType, None]:
        if not isinstance(other, PauliString):
            return NotImplemented
        return sum((not protocols.commutes(p0, p1) for p0, p1 in self.zip_paulis(other))) % 2 == 0

    def __neg__(self) -> 'PauliString':
        return PauliString(qubit_pauli_map=self._qubit_pauli_map, coefficient=-self._coefficient)

    def __pos__(self) -> 'PauliString':
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Override behavior of numpy's exp method."""
        if ufunc == np.exp and len(inputs) == 1 and (inputs[0] is self):
            return math.e ** self
        return NotImplemented

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return PauliString(qubit_pauli_map=self._qubit_pauli_map, coefficient=self.coefficient ** (-1))
        if self._is_parameterized_():
            return NotImplemented
        if isinstance(power, (int, float)):
            r, i = cmath.polar(self.coefficient)
            if abs(r - 1) > 0.0001:
                return NotImplemented
            if len(self) == 1:
                q, p = next(iter(self.items()))
                gates = {pauli_gates.X: common_gates.XPowGate, pauli_gates.Y: common_gates.YPowGate, pauli_gates.Z: common_gates.ZPowGate}
                return gates[p](exponent=power).on(q)
            global_half_turns = power * (i / math.pi)
            from cirq.ops import pauli_string_phasor
            return pauli_string_phasor.PauliStringPhasor(PauliString(qubit_pauli_map=self._qubit_pauli_map), exponent_neg=global_half_turns + power, exponent_pos=global_half_turns)
        return NotImplemented

    def __rpow__(self, base):
        if self._is_parameterized_():
            return NotImplemented
        if isinstance(base, (int, float)) and base > 0:
            if abs(self.coefficient.real) > 0.0001:
                raise NotImplementedError(f'Exponentiated to a non-Hermitian PauliString <{base}**{self}>. Coefficient must be imaginary.')
            half_turns = 2 * math.log(base) * (-self.coefficient.imag / math.pi)
            if len(self) == 1:
                q, p = next(iter(self.items()))
                gates = {pauli_gates.X: common_gates.XPowGate, pauli_gates.Y: common_gates.YPowGate, pauli_gates.Z: common_gates.ZPowGate}
                return gates[p](exponent=half_turns, global_shift=-0.5).on(q)
            from cirq.ops import pauli_string_phasor
            return pauli_string_phasor.PauliStringPhasor(PauliString(qubit_pauli_map=self._qubit_pauli_map), exponent_neg=+half_turns / 2, exponent_pos=-half_turns / 2)
        return NotImplemented

    def map_qubits(self, qubit_map: Dict[TKey, TKeyNew]) -> 'cirq.PauliString[TKeyNew]':
        """Replaces every qubit `q` in `self.qubits` with `qubit_map[q]`.

        Args:
            qubit_map: A map from qubits in the pauli string to new qubits.

        Returns:
            A new `PauliString` with remapped qubits.

        Raises:
            ValueError: If the map does not contain an entry for all qubits in the pauli string.
        """
        if not set(self.qubits) <= qubit_map.keys():
            raise ValueError(f"qubit_map must have a key for every qubit in the pauli strings' qubits. keys: {qubit_map.keys()} pauli string qubits: {self.qubits}")
        new_qubit_pauli_map = {qubit_map[qubit]: pauli for qubit, pauli in self.items()}
        return PauliString(qubit_pauli_map=new_qubit_pauli_map, coefficient=self._coefficient)

    def to_z_basis_ops(self) -> Iterator[raw_types.Operation]:
        """Returns single qubit operations to convert the qubits to the computational basis."""
        for qubit, pauli in self.items():
            yield clifford_gate.SingleQubitCliffordGate.from_single_map({pauli: (pauli_gates.Z, False)})(qubit)

    def dense(self, qubits: Sequence[TKey]) -> 'cirq.DensePauliString':
        """Returns a `cirq.DensePauliString` version of this Pauli string.

        This method satisfies the invariant `P.dense(qubits).on(*qubits) == P`.

        Args:
            qubits: The implicit sequence of qubits used by the dense pauli
                string. Specifically, if the returned dense Pauli string was
                applied to these qubits (via its `on` method) then the result
                would be a Pauli string equivalent to the receiving Pauli
                string.

        Returns:
            A `cirq.DensePauliString` instance `D` such that `D.on(*qubits)`
            equals the receiving `cirq.PauliString` instance `P`.

        Raises:
            ValueError: If the number of qubits is too small.
        """
        from cirq.ops.dense_pauli_string import DensePauliString
        if not self.keys() <= set(qubits):
            raise ValueError('not self.keys() <= set(qubits)')
        pauli_mask = [self.get(q, identity.I) for q in qubits]
        return DensePauliString(pauli_mask, coefficient=self.coefficient)

    def conjugated_by(self, clifford: 'cirq.OP_TREE') -> 'PauliString':
        """Returns the Pauli string conjugated by a clifford operation.

        The product-of-Paulis $P$ conjugated by the Clifford operation $C$ is

            $$
            C^\\dagger P C
            $$

        For example, conjugating a +Y operation by an S operation results in a
        +X operation (as opposed to a -X operation).

        In a circuit diagram where `P` is a pauli string observable immediately
        after a Clifford operation `C`, the pauli string `P.conjugated_by(C)` is
        the equivalent pauli string observable just before `C`.

            --------------------------C---P---

            = ---C---P------------------------

            = ---C---P---------C^-1---C-------

            = ---C---P---C^-1---------C-------

            = --(C^-1 · P · C)--------C-------

            = ---P.conjugated_by(C)---C-------

        Analogously, a Pauli product P can be moved from before a Clifford C in
        a circuit diagram to after the Clifford C by conjugating P by the
        inverse of C:

            ---P---C---------------------------

            = -----C---P.conjugated_by(C^-1)---

        Args:
            clifford: The Clifford operation to conjugate by. This can be an
                individual operation, or a tree of operations.

                Note that the composite Clifford operation defined by a sequence
                of operations is equivalent to a circuit containing those
                operations in the given order. Somewhat counter-intuitively,
                this means that the operations in the sequence are conjugated
                onto the Pauli string in reverse order. For example,
                `P.conjugated_by([C1, C2])` is equivalent to
                `P.conjugated_by(C2).conjugated_by(C1)`.

        Examples:
            >>> a, b = cirq.LineQubit.range(2)
            >>> print(cirq.X(a).conjugated_by(cirq.CZ(a, b)))
            X(q(0))*Z(q(1))
            >>> print(cirq.X(a).conjugated_by(cirq.S(a)))
            -Y(q(0))
            >>> print(cirq.X(a).conjugated_by([cirq.H(a), cirq.CNOT(a, b)]))
            Z(q(0))*X(q(1))

        Returns:
            The Pauli string conjugated by the given Clifford operation.
        """
        pauli_map = dict(self._qubit_pauli_map)
        should_negate = False
        for op in list(op_tree.flatten_to_ops(clifford))[::-1]:
            if pauli_map.keys().isdisjoint(set(op.qubits)):
                continue
            for clifford_op in _decompose_into_cliffords(op)[::-1]:
                if pauli_map.keys().isdisjoint(set(clifford_op.qubits)):
                    continue
                should_negate ^= _pass_operation_over(pauli_map, clifford_op, False)
        coef = -self._coefficient if should_negate else self.coefficient
        return PauliString(qubit_pauli_map=pauli_map, coefficient=coef)

    def after(self, ops: 'cirq.OP_TREE') -> 'cirq.PauliString':
        """Determines the equivalent pauli string after some operations.

        If the PauliString is $P$ and the Clifford operation is $C$, then the
        result is $C P C^\\dagger$.

        Args:
            ops: A stabilizer operation or nested collection of stabilizer
                operations.

        Returns:
            The result of propagating this pauli string from before to after the
            given operations.
        """
        return self.conjugated_by(protocols.inverse(ops))

    def before(self, ops: 'cirq.OP_TREE') -> 'cirq.PauliString':
        """Determines the equivalent pauli string before some operations.

        If the PauliString is $P$ and the Clifford operation is $C$, then the
        result is $C^\\dagger P C$.

        Args:
            ops: A stabilizer operation or nested collection of stabilizer
                operations.

        Returns:
            The result of propagating this pauli string from after to before the
            given operations.
        """
        return self.conjugated_by(ops)

    def pass_operations_over(self, ops: Iterable['cirq.Operation'], after_to_before: bool=False) -> 'PauliString':
        """Determines how the Pauli string changes when conjugated by Cliffords.

        The output and input pauli strings are related by a circuit equivalence.
        In particular, this circuit:

            ───ops───INPUT_PAULI_STRING───

        will be equivalent to this circuit:

            ───OUTPUT_PAULI_STRING───ops───

        up to global phase (assuming `after_to_before` is not set).

        If ops together have matrix C, the Pauli string has matrix P, and the
        output Pauli string has matrix P', then P' == C^-1 P C up to
        global phase.

        Setting `after_to_before` inverts the relationship, so that the output
        is the input and the input is the output. Equivalently, it inverts C.

        Args:
            ops: The operations to move over the string.
            after_to_before: Determines whether the operations start after the
                pauli string, instead of before (and so are moving in the
                opposite direction).
        """
        pauli_map = dict(self._qubit_pauli_map)
        should_negate = False
        for op in ops:
            if pauli_map.keys().isdisjoint(set(op.qubits)):
                continue
            decomposed = _decompose_into_cliffords(op)
            if not after_to_before:
                decomposed = decomposed[::-1]
            for clifford_op in decomposed:
                if pauli_map.keys().isdisjoint(set(clifford_op.qubits)):
                    continue
                should_negate ^= _pass_operation_over(pauli_map, clifford_op, after_to_before)
        coef = -self._coefficient if should_negate else self.coefficient
        return PauliString(qubit_pauli_map=pauli_map, coefficient=coef)

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.coefficient)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.coefficient)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'cirq.PauliString':
        coefficient = protocols.resolve_parameters(self.coefficient, resolver, recursive)
        return PauliString(qubit_pauli_map=self._qubit_pauli_map, coefficient=coefficient)