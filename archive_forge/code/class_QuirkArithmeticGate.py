import inspect
from typing import (
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES
@value.value_equality
class QuirkArithmeticGate(ops.ArithmeticGate):
    """Applies arithmetic to a target and some inputs.

    Implements Quirk-specific implicit effects like assuming that the presence
    of an 'r' input implies modular arithmetic.

    In Quirk, modular operations have no effect on values larger than the
    modulus. This convention is used because unitarity forces *some* convention
    on out-of-range values (they cannot simply disappear or raise exceptions),
    and the simplest is to do nothing. This call handles ensuring that happens,
    and ensuring the new target register value is normalized modulo the modulus.
    """

    def __init__(self, identifier: str, target: Sequence[int], inputs: Sequence[Union[Sequence[int], int]]):
        """Inits QuirkArithmeticGate.

        Args:
            identifier: The quirk identifier string for this operation.
            target: The target qubit register.
            inputs: Qubit registers, which correspond to the qid shape of the
                qubits from which the input will be read, or classical
                constants, that determine what happens to the target.

        Raises:
            ValueError: If the target is too small for a modular operation with
                too small modulus.
        """
        self.identifier = identifier
        self.target: Tuple[int, ...] = tuple(target)
        self.inputs: Tuple[Union[Sequence[int], int], ...] = tuple((e if isinstance(e, int) else tuple(e) for e in inputs))
        if self.operation.is_modular:
            r = inputs[-1]
            if isinstance(r, int):
                over = r > 1 << len(target)
            else:
                over = len(cast(Sequence, r)) > len(target)
            if over:
                raise ValueError(f'Target too small for modulus.\nTarget: {target}\nModulus: {r}')

    @property
    def operation(self) -> '_QuirkArithmeticCallable':
        return ARITHMETIC_OP_TABLE[self.identifier]

    def _value_equality_values_(self) -> Any:
        return (self.identifier, self.target, self.inputs)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [self.target, *self.inputs]

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> 'QuirkArithmeticGate':
        if len(new_registers) != len(self.inputs) + 1:
            raise ValueError(f'Wrong number of registers.\nNew registers: {repr(new_registers)}\nOperation: {repr(self)}')
        if isinstance(new_registers[0], int):
            raise ValueError(f'The first register is the mutable target. It must be a list of qubits, not the constant {new_registers[0]}.')
        return QuirkArithmeticGate(self.identifier, new_registers[0], new_registers[1:])

    def apply(self, *registers: int) -> Union[int, Iterable[int]]:
        return self.operation(*registers)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> List[str]:
        lettered_args = list(zip(self.operation.letters, self.inputs))
        result: List[str] = []
        consts = ''.join((f',{letter}={reg}' for letter, reg in lettered_args if isinstance(reg, int)))
        result.append(f'Quirk({self.identifier}{consts})')
        result.extend((f'#{i}' for i in range(2, len(self.target) + 1)))
        for letter, reg in lettered_args:
            if not isinstance(reg, int):
                result.extend((f'{letter.upper()}{i}' for i in range(len(cast(Sequence, reg)))))
        return result

    def __repr__(self) -> str:
        return f'cirq.interop.quirk.QuirkArithmeticGate(\n    {repr(self.identifier)},\n    target={repr(self.target)},\n    inputs={_indented_list_lines_repr(self.inputs)},\n)'