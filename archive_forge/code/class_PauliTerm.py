import re
from itertools import product
import numpy as np
import copy
from typing import (
from pyquil.quilatom import (
from .quil import Program
from .gates import H, RZ, RX, CNOT, X, PHASE, QUANTUM_GATES
from numbers import Number, Complex
from collections import OrderedDict
import warnings
class PauliTerm(object):
    """A term is a product of Pauli operators operating on different qubits."""

    def __init__(self, op: str, index: Optional[PauliTargetDesignator], coefficient: ExpressionDesignator=1.0):
        """Create a new Pauli Term with a Pauli operator at a particular index and a leading
        coefficient.

        :param op: The Pauli operator as a string "X", "Y", "Z", or "I"
        :param index: The qubit index that that operator is applied to.
        :param coefficient: The coefficient multiplying the operator, e.g. 1.5 * Z_1
        """
        if op not in PAULI_OPS:
            raise ValueError(f'{op} is not a valid Pauli operator')
        self._ops: Dict[PauliTargetDesignator, str] = OrderedDict()
        if op != 'I':
            if not _valid_qubit(index):
                raise ValueError(f'{index} is not a valid qubit')
            assert index is not None
            self._ops[index] = op
        if isinstance(coefficient, Number):
            self.coefficient: Union[complex, Expression] = complex(coefficient)
        else:
            self.coefficient = coefficient

    def id(self, sort_ops: bool=True) -> str:
        """
        Returns an identifier string for the PauliTerm (ignoring the coefficient).

        Don't use this to compare terms. This function will not work with qubits that
        aren't sortable.

        :param sort_ops: Whether to sort operations by qubit. This is True by default for
            backwards compatibility but will change in a future version. Callers should never rely
            on comparing id's for testing equality. See ``operations_as_set`` instead.
        :return: A string representation of this term's operations.
        """
        if len(self._ops) == 0 and (not sort_ops):
            return 'I'
        if sort_ops and len(self._ops) > 1:
            warnings.warn('`PauliTerm.id()` will not work on PauliTerms where the qubits are not sortable and should be avoided in favor of `operations_as_set`.', FutureWarning)
            return ''.join(('{}{}'.format(self._ops[q], q) for q in sorted(self._ops.keys())))
        else:
            return ''.join(('{}{}'.format(p, q) for q, p in self._ops.items()))

    def operations_as_set(self) -> FrozenSet[Tuple[PauliTargetDesignator, str]]:
        """
        Return a frozenset of operations in this term.

        Use this in place of :py:func:`id` if the order of operations in the term does not
        matter.

        :return: frozenset of (qubit, op_str) representing Pauli operations
        """
        return frozenset(self._ops.items())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (PauliTerm, PauliSum)):
            raise TypeError("Can't compare PauliTerm with object of type {}.".format(type(other)))
        elif isinstance(other, PauliSum):
            return other == self
        else:
            return self.operations_as_set() == other.operations_as_set() and np.allclose(self.coefficient, other.coefficient)

    def __hash__(self) -> int:
        assert isinstance(self.coefficient, Complex)
        return hash((round(self.coefficient.real * HASH_PRECISION), round(self.coefficient.imag * HASH_PRECISION), self.operations_as_set()))

    def __len__(self) -> int:
        """
        The length of the PauliTerm is the number of Pauli operators in the term. A term that
        consists of only a scalar has a length of zero.
        """
        return len(self._ops)

    def copy(self) -> 'PauliTerm':
        """
        Properly creates a new PauliTerm, with a completely new dictionary
        of operators
        """
        new_term = PauliTerm('I', 0, 1.0)
        for key in self.__dict__.keys():
            val = self.__dict__[key]
            if isinstance(val, (dict, list, set)):
                new_term.__dict__[key] = copy.copy(val)
            else:
                new_term.__dict__[key] = val
        return new_term

    @property
    def program(self) -> Program:
        return Program([QUANTUM_GATES[gate](q) for q, gate in self])

    def get_qubits(self) -> List[PauliTargetDesignator]:
        """Gets all the qubits that this PauliTerm operates on."""
        return list(self._ops.keys())

    def __getitem__(self, i: PauliTargetDesignator) -> str:
        return self._ops.get(i, 'I')

    def __iter__(self) -> Iterator[Tuple[PauliTargetDesignator, str]]:
        for i in self.get_qubits():
            yield (i, self[i])

    def _multiply_factor(self, factor: str, index: PauliTargetDesignator) -> 'PauliTerm':
        new_term = PauliTerm('I', 0)
        new_coeff = self.coefficient
        new_ops = self._ops.copy()
        ops = self[index] + factor
        new_op = PAULI_PROD[ops]
        if new_op != 'I':
            new_ops[index] = new_op
        else:
            del new_ops[index]
        new_coeff *= PAULI_COEFF[ops]
        new_term._ops = new_ops
        new_term.coefficient = new_coeff
        return new_term

    def __mul__(self, term: Union[PauliDesignator, ExpressionDesignator]) -> PauliDesignator:
        """Multiplies this Pauli Term with another PauliTerm, PauliSum, or number according to the
        Pauli algebra rules.

        :param term: (PauliTerm or PauliSum or Number) A term to multiply by.
        :returns: The product of this PauliTerm and term.
        """
        if isinstance(term, PauliSum):
            return (PauliSum([self]) * term).simplify()
        elif isinstance(term, PauliTerm):
            new_term = PauliTerm('I', 0, 1.0)
            new_term._ops = self._ops.copy()
            new_coeff = self.coefficient * term.coefficient
            for index, op in term:
                new_term = new_term._multiply_factor(op, index)
            return term_with_coeff(new_term, new_term.coefficient * new_coeff)
        return term_with_coeff(self, self.coefficient * term)

    def __rmul__(self, other: ExpressionDesignator) -> 'PauliTerm':
        """Multiplies this PauliTerm with another object, probably a number.

        :param other: A number or PauliTerm to multiply by
        :returns: A new PauliTerm
        """
        p = self * other
        assert isinstance(p, PauliTerm)
        return p

    def __pow__(self, power: int) -> 'PauliTerm':
        """Raises this PauliTerm to power.

        :param power: The power to raise this PauliTerm to.
        :return: The power-fold product of power.
        """
        if not isinstance(power, int) or power < 0:
            raise ValueError('The power must be a non-negative integer.')
        if len(self.get_qubits()) == 0:
            return term_with_coeff(self, 1)
        result = ID()
        for _ in range(power):
            result = cast(PauliTerm, result * self)
        return result

    def __add__(self, other: Union[PauliDesignator, ExpressionDesignator]) -> 'PauliSum':
        """Adds this PauliTerm with another one.

        :param other: A PauliTerm object, a PauliSum object, or a Number
        :returns: A PauliSum object representing the sum of this PauliTerm and other
        """
        if isinstance(other, PauliSum):
            return other + self
        elif isinstance(other, PauliTerm):
            new_sum = PauliSum([self, other])
            return new_sum.simplify()
        else:
            return self + PauliTerm('I', 0, other)

    def __radd__(self, other: ExpressionDesignator) -> 'PauliSum':
        """Adds this PauliTerm with a Number.

        :param other: A Number
        :returns: A new PauliSum
        """
        return PauliTerm('I', 0, other) + self

    def __sub__(self, other: Union['PauliTerm', ExpressionDesignator]) -> 'PauliSum':
        """Subtracts a PauliTerm from this one.

        :param other: A PauliTerm object, a number, or an Expression
        :returns: A PauliSum object representing the difference of this PauliTerm and term
        """
        return self + -1.0 * other

    def __rsub__(self, other: Union['PauliTerm', ExpressionDesignator]) -> 'PauliSum':
        """Subtracts this PauliTerm from a Number or PauliTerm.

        :param other: A PauliTerm object or a Number
        :returns: A PauliSum object representing the difference of this PauliTerm and term
        """
        return other + -1.0 * self

    def __repr__(self) -> str:
        term_strs = []
        for index in self._ops.keys():
            term_strs.append('%s%s' % (self[index], index))
        if len(term_strs) == 0:
            term_strs.append('I')
        out = '%s*%s' % (self.coefficient, '*'.join(term_strs))
        return out

    def compact_str(self) -> str:
        """A string representation of the Pauli term that is more compact than ``str(term)``

        >>> term = 2.0 * sX(1)* sZ(2)
        >>> str(term)
        >>> '2.0*X1*X2'
        >>> term.compact_str()
        >>> '2.0*X1X2'
        """
        return f'{self.coefficient}*{self.id(sort_ops=False)}'

    @classmethod
    def from_list(cls, terms_list: List[Tuple[str, int]], coefficient: float=1.0) -> 'PauliTerm':
        """
        Allocates a Pauli Term from a list of operators and indices. This is more efficient than
        multiplying together individual terms.

        :param list terms_list: A list of tuples, e.g. [("X", 0), ("Y", 1)]
        :return: PauliTerm
        """
        if not all([isinstance(op, tuple) for op in terms_list]):
            raise TypeError('The type of terms_list should be a list of (name, index) tuples suitable for PauliTerm().')
        pterm = PauliTerm('I', 0)
        assert all([op[0] in PAULI_OPS for op in terms_list])
        indices = [op[1] for op in terms_list]
        assert all((_valid_qubit(index) for index in indices))
        if len(set(indices)) != len(indices):
            raise ValueError('Elements of PauliTerm that are allocated using from_list must be on disjoint qubits. Use PauliTerm multiplication to simplify terms instead.')
        for op, index in terms_list:
            if op != 'I':
                pterm._ops[index] = op
        if isinstance(coefficient, Number):
            pterm.coefficient = complex(coefficient)
        else:
            pterm.coefficient = coefficient
        return pterm

    @classmethod
    def from_compact_str(cls, str_pauli_term: str) -> 'PauliTerm':
        """Construct a PauliTerm from the result of str(pauli_term)"""
        try:
            str_coef, str_op = re.split('\\*(?![^(]*\\))', str_pauli_term, maxsplit=1)
        except ValueError:
            raise ValueError(f'Could not separate the pauli string into coefficient and operator. {str_pauli_term} does not match <coefficient>*<operator>')
        str_coef = str_coef.replace(' ', '')
        try:
            coef: Union[float, complex] = float(str_coef)
        except ValueError:
            try:
                coef = complex(str_coef)
            except ValueError:
                raise ValueError(f'Could not parse the coefficient {str_coef}')
        op = sI() * coef
        if str_op == 'I':
            assert isinstance(op, PauliTerm)
            return op
        str_op = re.sub('\\*', '', str_op)
        if not re.match('^(([XYZ])(\\d+))+$', str_op):
            raise ValueError(f'Could not parse operator string {str_op}. It should match ^(([XYZ])(\\d+))+$')
        for factor in re.finditer('([XYZ])(\\d+)', str_op):
            op *= cls(factor.group(1), int(factor.group(2)))
        assert isinstance(op, PauliTerm)
        return op

    def pauli_string(self, qubits: Iterable[int]) -> str:
        """
        Return a string representation of this PauliTerm without its coefficient and with
        implicit qubit indices.

        If an iterable of qubits is provided, each character in the resulting string represents
        a Pauli operator on the corresponding qubit.

        >>> p = PauliTerm("X", 0) * PauliTerm("Y", 1, 1.j)
        >>> p.pauli_string()
        "XY"
        >>> p.pauli_string(qubits=[0])
        "X"
        >>> p.pauli_string(qubits=[0, 2])
        "XI"

        :param iterable of qubits: The iterable of qubits to represent, given as ints.
        :return: The string representation of this PauliTerm, sans coefficient
        """
        return ''.join((self[q] for q in qubits))