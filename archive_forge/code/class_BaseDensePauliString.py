import abc
import numbers
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, linalg, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, identity, pauli_gates, global_phase_op, pauli_string
from cirq.type_workarounds import NotImplementedType
@value.value_equality(approximate=True, distinct_child_types=True)
class BaseDensePauliString(raw_types.Gate, metaclass=abc.ABCMeta):
    """Parent class for `cirq.DensePauliString` and `cirq.MutableDensePauliString`.

    `cirq.BaseDensePauliString` is an abstract base class, which is used to implement
    `cirq.DensePauliString` and `cirq.MutableDensePauliString`. The non-mutable version
    is used as the corresponding gate for `cirq.PauliString` operation and the mutable
    version is mainly used for efficiently manipulating dense pauli strings.

    See the docstrings of `cirq.DensePauliString` and `cirq.MutableDensePauliString` for more
    details.

    Examples:
    >>> print(cirq.DensePauliString('XXIY'))
    +XXIY

    >>> print(cirq.MutableDensePauliString('IZII', coefficient=-1))
    -IZII (mutable)

    >>> print(cirq.DensePauliString([0, 1, 2, 3],
    ...                             coefficient=sympy.Symbol('t')))
    t*IXYZ
    """
    I_VAL = 0
    X_VAL = 1
    Y_VAL = 2
    Z_VAL = 3

    def __init__(self, pauli_mask: Union[Iterable['cirq.PAULI_GATE_LIKE'], np.ndarray], *, coefficient: 'cirq.TParamValComplex'=1):
        """Initializes a new dense pauli string.

        Args:
            pauli_mask: A specification of the Pauli gates to use. This argument
                can be a string like "IXYYZ", or a numeric list like
                [0, 1, 3, 2] with I=0, X=1, Y=2, Z=3=X|Y.

                The internal representation is a 1-dimensional uint8 numpy array
                containing numeric values. If such a numpy array is given, and
                the pauli string is mutable, the argument will be used directly
                instead of being copied.
            coefficient: A complex number. Usually +1, -1, 1j, or -1j but other
                values are supported.
        """
        self._pauli_mask = _as_pauli_mask(pauli_mask)
        self._coefficient: Union[complex, sympy.Expr] = coefficient if isinstance(coefficient, sympy.Expr) else complex(coefficient)
        if type(self) != MutableDensePauliString:
            self._pauli_mask = np.copy(self.pauli_mask)
            self._pauli_mask.flags.writeable = False

    @property
    def pauli_mask(self) -> np.ndarray:
        """A 1-dimensional uint8 numpy array giving a specification of Pauli gates to use."""
        return self._pauli_mask

    @property
    def coefficient(self) -> Union[sympy.Expr, complex]:
        """A complex coefficient or symbol."""
        return self._coefficient

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['pauli_mask', 'coefficient'])

    def _value_equality_values_(self):
        return (self.coefficient, tuple((PAULI_CHARS[p] for p in self.pauli_mask)))

    @classmethod
    def one_hot(cls, *, index: int, length: int, pauli: 'cirq.PAULI_GATE_LIKE') -> Self:
        """Creates a dense pauli string with only one non-identity Pauli.

        Args:
            index: The index of the Pauli that is not an identity.
            length: The total length of the string to create.
            pauli: The pauli gate to put at the hot index. Can be set to either
                a string ('X', 'Y', 'Z', 'I'), a cirq gate (`cirq.X`,
                `cirq.Y`, `cirq.Z`, or `cirq.I`), or an integer (0=I, 1=X, 2=Y,
                3=Z).
        """
        mask = np.zeros(length, dtype=np.uint8)
        mask[index] = _pauli_index(pauli)
        concrete_cls = cast(Callable, DensePauliString if cls is BaseDensePauliString else cls)
        return concrete_cls(pauli_mask=mask)

    @classmethod
    def eye(cls, length: int) -> Self:
        """Creates a dense pauli string containing only identity gates.

        Args:
            length: The length of the dense pauli string.
        """
        concrete_cls = cast(Callable, DensePauliString if cls is BaseDensePauliString else cls)
        return concrete_cls(pauli_mask=np.zeros(length, dtype=np.uint8))

    def _num_qubits_(self) -> int:
        return len(self)

    def _has_unitary_(self) -> bool:
        if self._is_parameterized_():
            return False
        return abs(1 - abs(cast(complex, self.coefficient))) < 1e-08

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        if not self._has_unitary_():
            return NotImplemented
        return self.coefficient * linalg.kron(*[protocols.unitary(PAULI_GATES[p]) for p in self.pauli_mask])

    def _apply_unitary_(self, args) -> Union[np.ndarray, None, NotImplementedType]:
        if not self._has_unitary_():
            return NotImplemented
        from cirq import devices
        qubits = devices.LineQubit.range(len(self))
        decomposed_ops = cast(Iterable['cirq.OP_TREE'], self._decompose_(qubits))
        return protocols.apply_unitaries(decomposed_ops, qubits, args)

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> Union[NotImplementedType, 'cirq.OP_TREE']:
        if not self._has_unitary_():
            return NotImplemented
        result = [PAULI_GATES[p].on(q) for p, q in zip(self.pauli_mask, qubits) if p]
        if self.coefficient != 1:
            result.append(global_phase_op.global_phase_operation(self.coefficient))
        return result

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.coefficient)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.coefficient)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> Self:
        return self.copy(coefficient=protocols.resolve_parameters(self.coefficient, resolver, recursive))

    def __pos__(self):
        return self

    def __pow__(self, power: Union[int, float]) -> Union[NotImplementedType, Self]:
        concrete_class = type(self)
        if isinstance(power, int):
            i_group = [1, +1j, -1, -1j]
            coef = i_group[i_group.index(cast(complex, self.coefficient)) * power % 4] if self.coefficient in i_group else self.coefficient ** power
            if power % 2 == 0:
                return concrete_class.eye(len(self)).__mul__(coef)
            return concrete_class(coefficient=coef, pauli_mask=self.pauli_mask)
        return NotImplemented

    @overload
    def __getitem__(self, item: int) -> Union['cirq.Pauli', 'cirq.IdentityGate']:
        pass

    @overload
    def __getitem__(self, item: slice) -> Self:
        pass

    def __getitem__(self, item):
        if isinstance(item, int):
            return PAULI_GATES[self.pauli_mask[item]]
        if isinstance(item, slice):
            return type(self)(coefficient=1, pauli_mask=self.pauli_mask[item])
        raise TypeError(f'indices must be integers or slices, not {type(item)}')

    def __iter__(self) -> Iterator[Union['cirq.Pauli', 'cirq.IdentityGate']]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self.pauli_mask)

    def __neg__(self):
        return type(self)(coefficient=-self.coefficient, pauli_mask=self.pauli_mask)

    def __truediv__(self, other):
        if isinstance(other, (sympy.Basic, numbers.Number)):
            return self.__mul__(1 / other)
        return NotImplemented

    def __mul__(self, other):
        concrete_class = type(self)
        if isinstance(other, BaseDensePauliString):
            if isinstance(other, MutableDensePauliString):
                concrete_class = MutableDensePauliString
            max_len = max(len(self.pauli_mask), len(other.pauli_mask))
            min_len = min(len(self.pauli_mask), len(other.pauli_mask))
            new_mask = np.zeros(max_len, dtype=np.uint8)
            new_mask[:len(self.pauli_mask)] ^= self.pauli_mask
            new_mask[:len(other.pauli_mask)] ^= other.pauli_mask
            tweak = _vectorized_pauli_mul_phase(self.pauli_mask[:min_len], other.pauli_mask[:min_len])
            return concrete_class(pauli_mask=new_mask, coefficient=self.coefficient * other.coefficient * tweak)
        if isinstance(other, (sympy.Basic, numbers.Number)):
            new_coef = protocols.mul(self.coefficient, other, default=None)
            if new_coef is None:
                return NotImplemented
            return concrete_class(pauli_mask=self.pauli_mask, coefficient=new_coef)
        split = _attempt_value_to_pauli_index(other)
        if split is not None:
            p, i = split
            mask = np.copy(self.pauli_mask)
            mask[i] ^= p
            return concrete_class(pauli_mask=mask, coefficient=self.coefficient * _vectorized_pauli_mul_phase(self.pauli_mask[i], p))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (sympy.Basic, numbers.Number)):
            return self.__mul__(other)
        split = _attempt_value_to_pauli_index(other)
        if split is not None:
            p, i = split
            mask = np.copy(self.pauli_mask)
            mask[i] ^= p
            return type(self)(pauli_mask=mask, coefficient=self.coefficient * _vectorized_pauli_mul_phase(p, self.pauli_mask[i]))
        return NotImplemented

    def tensor_product(self, other: 'BaseDensePauliString') -> Self:
        """Concatenates dense pauli strings and multiplies their coefficients.

        Args:
            other: The dense pauli string to place after the end of this one.

        Returns:
            A dense pauli string with the concatenation of the paulis from the
            two input pauli strings, and the product of their coefficients.
        """
        return type(self)(coefficient=self.coefficient * other.coefficient, pauli_mask=np.concatenate([self.pauli_mask, other.pauli_mask]))

    def __abs__(self) -> Self:
        coef = self.coefficient
        return type(self)(coefficient=sympy.Abs(coef) if isinstance(coef, sympy.Expr) else abs(coef), pauli_mask=self.pauli_mask)

    def on(self, *qubits: 'cirq.Qid') -> 'cirq.PauliString':
        return self.sparse(qubits)

    def sparse(self, qubits: Optional[Sequence['cirq.Qid']]=None) -> 'cirq.PauliString':
        """A `cirq.PauliString` version of this dense pauli string.

        Args:
            qubits: The qubits to apply the Paulis to. Defaults to
                `cirq.LineQubit.range(len(self))`.

        Returns:
            A `cirq.PauliString` with the non-identity operations from
            this dense pauli string applied to appropriate qubits.

        Raises:
            ValueError: If the number of qubits supplied does not match that of
                this instance.
        """
        if qubits is None:
            from cirq import devices
            qubits = devices.LineQubit.range(len(self))
        if len(qubits) != len(self):
            raise ValueError('Wrong number of qubits.')
        return pauli_string.PauliString(coefficient=self.coefficient, qubit_pauli_map={q: PAULI_GATES[p] for q, p in zip(qubits, self.pauli_mask) if p})

    def __str__(self) -> str:
        if self.coefficient == 1:
            coef = '+'
        elif self.coefficient == -1:
            coef = '-'
        elif isinstance(self.coefficient, (complex, sympy.Symbol)):
            coef = f'{self.coefficient}*'
        else:
            coef = f'({self.coefficient})*'
        mask = ''.join((PAULI_CHARS[p] for p in self.pauli_mask))
        return coef + mask

    def __repr__(self) -> str:
        paulis = ''.join((PAULI_CHARS[p] for p in self.pauli_mask))
        return f'cirq.{type(self).__name__}({repr(paulis)}, coefficient={proper_repr(self.coefficient)})'

    def _commutes_(self, other: Any, *, atol: float=1e-08) -> Union[bool, NotImplementedType, None]:
        if isinstance(other, BaseDensePauliString):
            n = min(len(self.pauli_mask), len(other.pauli_mask))
            phase = _vectorized_pauli_mul_phase(self.pauli_mask[:n], other.pauli_mask[:n])
            return phase == 1 or phase == -1
        split = _attempt_value_to_pauli_index(other)
        if split is not None:
            p1, i = split
            p2 = self.pauli_mask[i]
            return (p1 or p2) == (p2 or p1)
        return NotImplemented

    def frozen(self) -> 'DensePauliString':
        """A `cirq.DensePauliString` with the same contents."""
        return DensePauliString(coefficient=self.coefficient, pauli_mask=self.pauli_mask)

    def mutable_copy(self) -> 'MutableDensePauliString':
        """A `cirq.MutableDensePauliString` with the same contents."""
        return MutableDensePauliString(coefficient=self.coefficient, pauli_mask=np.copy(self.pauli_mask))

    @abc.abstractmethod
    def copy(self, coefficient: Optional[Union[sympy.Expr, int, float, complex]]=None, pauli_mask: Union[None, str, Iterable[int], np.ndarray]=None) -> Self:
        """Returns a copy with possibly modified contents.

        Args:
            coefficient: The new coefficient value. If not specified, defaults
                to the current `coefficient` value.
            pauli_mask: The new `pauli_mask` value. If not specified, defaults
                to the current pauli mask value.

        Returns:
            A copied instance.
        """