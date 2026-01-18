import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
@dataclass(frozen=True, init=False)
class ExperimentSetting:
    """
    Input and output settings for a tomography-like experiment.

    Many near-term quantum algorithms take the following form:

     - Start in a pauli state
     - Prepare some ansatz
     - Measure it w.r.t. pauli operators

    Where we typically use a large number of (start, measure) pairs but keep the ansatz preparation
    program consistent. This class represents the (start, measure) pairs. Typically a large
    number of these :py:class:`ExperimentSetting` objects will be created and grouped into
    an :py:class:`Experiment`.

    :ivar additional_expectations: A list of lists, where each inner list specifies a qubit subset
        to calculate the joint expectation value for. This attribute allows users to extract
        simultaneously measurable expectation values from a single experiment setting.
    """
    in_state: TensorProductState
    out_operator: PauliTerm
    additional_expectations: Optional[List[List[int]]] = None

    def __init__(self, in_state: TensorProductState, out_operator: PauliTerm, additional_expectations: Optional[List[List[int]]]=None):
        object.__setattr__(self, 'in_state', in_state)
        object.__setattr__(self, 'out_operator', out_operator)
        object.__setattr__(self, 'additional_expectations', additional_expectations)

    def _in_operator(self) -> PauliTerm:
        pt = sI()
        for oneq_state in self.in_state.states:
            if oneq_state.label not in ['X', 'Y', 'Z']:
                raise ValueError(f"Can't shim {oneq_state.label} into a pauli term. Use in_state.")
            if oneq_state.index != 0:
                raise ValueError(f"Can't shim {oneq_state} into a pauli term. Use in_state.")
            new_pt = pt * PauliTerm(op=oneq_state.label, index=oneq_state.qubit)
            pt = cast(PauliTerm, new_pt)
        return pt

    def __str__(self) -> str:
        return f'{self.in_state}→{self.out_operator.compact_str()}'

    def __repr__(self) -> str:
        return f'ExperimentSetting[{self}]'

    def serializable(self) -> str:
        return str(self)

    @classmethod
    def from_str(cls, s: str) -> 'ExperimentSetting':
        """The opposite of str(expt)"""
        instr, outstr = s.split('→')
        return ExperimentSetting(in_state=TensorProductState.from_str(instr), out_operator=PauliTerm.from_compact_str(outstr))