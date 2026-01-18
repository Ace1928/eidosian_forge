import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
@value.value_equality
class CircuitDiagramInfo:
    """Describes how to draw an operation in a circuit diagram."""

    def __init__(self, wire_symbols: Iterable[str], exponent: Any=1, connected: bool=True, exponent_qubit_index: Optional[int]=None, auto_exponent_parens: bool=True) -> None:
        """Inits CircuitDiagramInfo.

        Args:
            wire_symbols: The symbols that should be shown on the qubits
                affected by this operation. Must match the number of qubits that
                the operation is applied to.
            exponent: An optional convenience value that will be appended onto
                an operation's final gate symbol with a caret in front
                (unless it's equal to 1). For example, the square root of X gate
                has a text diagram exponent of 0.5 and symbol of 'X' so it is
                drawn as 'X^0.5'.
            connected: Whether or not to draw a line connecting the qubits.
            exponent_qubit_index: The qubit to put the exponent on. (The k'th
                qubit is the k'th target of the gate.) Defaults to the bottom
                qubit in the diagram.
            auto_exponent_parens: When this is True, diagram making code will
                add parentheses around exponents whose contents could look
                ambiguous (e.g. if the exponent contains a dash character that
                could be mistaken for an identity wire). Defaults to True.

        Raises:
            ValueError: If `wire_symbols` is a string, and not an iterable
                of strings.
        """
        if isinstance(wire_symbols, str):
            raise ValueError('Expected an Iterable[str] for wire_symbols but got a str.')
        self.wire_symbols = tuple(wire_symbols)
        self.exponent = exponent
        self.connected = connected
        self.exponent_qubit_index = exponent_qubit_index
        self.auto_exponent_parens = auto_exponent_parens

    def with_wire_symbols(self, new_wire_symbols: Iterable[str]):
        return CircuitDiagramInfo(wire_symbols=new_wire_symbols, exponent=self.exponent, connected=self.connected, exponent_qubit_index=self.exponent_qubit_index, auto_exponent_parens=self.auto_exponent_parens)

    def _value_equality_values_(self) -> Any:
        return (self.wire_symbols, self.exponent, self.connected, self.exponent_qubit_index, self.auto_exponent_parens)

    def _wire_symbols_including_formatted_exponent(self, args: 'cirq.CircuitDiagramInfoArgs', *, preferred_exponent_index: Optional[int]=None) -> List[str]:
        result = list(self.wire_symbols)
        exponent = self._formatted_exponent(args)
        if exponent is not None:
            ks: Sequence[int]
            if self.exponent_qubit_index is not None:
                ks = (self.exponent_qubit_index,)
            elif not self.connected:
                ks = range(len(result))
            elif preferred_exponent_index is not None:
                ks = (preferred_exponent_index,)
            else:
                ks = (0,)
            for k in ks:
                result[k] += f'^{exponent}'
        return result

    def _formatted_exponent(self, args: 'cirq.CircuitDiagramInfoArgs') -> Optional[str]:
        if protocols.is_parameterized(self.exponent):
            name = str(self.exponent)
            return f'({name})' if _is_exposed_formula(name) else name
        if self.exponent == 0:
            return '0'
        if self.exponent == 1:
            return None
        if self.exponent == -1:
            return '-1'
        if isinstance(self.exponent, float):
            if args.precision is not None:
                approx_frac = Fraction(self.exponent).limit_denominator(16)
                if approx_frac.denominator not in [2, 4, 5, 10]:
                    if abs(float(approx_frac) - self.exponent) < 10 ** (-args.precision):
                        return f'({approx_frac})'
                return args.format_real(self.exponent)
            return repr(self.exponent)
        s = str(self.exponent)
        if self.auto_exponent_parens and ('+' in s or ' ' in s or '-' in s[1:]):
            return f'({self.exponent})'
        return s

    def __repr__(self) -> str:
        return f'cirq.CircuitDiagramInfo(wire_symbols={self.wire_symbols!r}, exponent={self.exponent!r}, connected={self.connected!r}, exponent_qubit_index={self.exponent_qubit_index!r}, auto_exponent_parens={self.auto_exponent_parens!r})'