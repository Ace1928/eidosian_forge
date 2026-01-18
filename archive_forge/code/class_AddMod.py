from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import attr
import cirq
from cirq._compat import cached_property
from numpy.typing import NDArray
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_class
@deprecated_cirq_ft_class()
@attr.frozen(auto_attribs=True)
class AddMod(cirq.ArithmeticGate):
    """Applies U_{M}_{add}|x> = |(x + add) % M> if x < M else |x>.

    Applies modular addition to input register `|x>` given parameters `mod` and `add_val` s.t.
        1) If integer `x` < `mod`: output is `|(x + add) % M>`
        2) If integer `x` >= `mod`: output is `|x>`.

    This condition is needed to ensure that the mapping of all input basis states (i.e. input
    states |0>, |1>, ..., |2 ** bitsize - 1) to corresponding output states is bijective and thus
    the gate is reversible.

    Also supports controlled version of the gate by specifying a per qubit control value as a tuple
    of integers passed as `cv`.
    """
    bitsize: int
    mod: int = attr.field()
    add_val: int = 1
    cv: Tuple[int, ...] = attr.field(converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=())

    @mod.validator
    def _validate_mod(self, attribute, value):
        if not 1 <= value <= 2 ** self.bitsize:
            raise ValueError(f'mod: {value} must be between [1, {2 ** self.bitsize}].')

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        add_reg = (2,) * self.bitsize
        control_reg = (2,) * len(self.cv)
        return (control_reg, add_reg) if control_reg else (add_reg,)

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> 'AddMod':
        raise NotImplementedError()

    def apply(self, *args) -> Union[int, Iterable[int]]:
        target_val = args[-1]
        if target_val < self.mod:
            new_target_val = (target_val + self.add_val) % self.mod
        else:
            new_target_val = target_val
        if self.cv and args[0] != int(''.join((str(x) for x in self.cv)), 2):
            new_target_val = target_val
        ret = (args[0], new_target_val) if self.cv else (new_target_val,)
        return ret

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if b else '@(0)' for b in self.cv]
        wire_symbols += [f'Add_{self.add_val}_Mod_{self.mod}'] * self.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int) -> 'AddMod':
        return AddMod(self.bitsize, self.mod, add_val=self.add_val * power, cv=self.cv)

    def __repr__(self) -> str:
        return f'cirq_ft.AddMod({self.bitsize}, {self.mod}, {self.add_val}, {self.cv})'

    def _t_complexity_(self) -> infra.TComplexity:
        return 5 * infra.t_complexity(AdditionGate(self.bitsize))