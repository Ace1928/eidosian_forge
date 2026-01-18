import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, Type, TypeVar, Union
class FsmWithContext(Generic[T_FsmInputs, T_FsmContext]):
    _state_dict: Dict[Type[FsmState], FsmState]
    _table: FsmTableWithContext[T_FsmInputs, T_FsmContext]
    _state: FsmState[T_FsmInputs, T_FsmContext]
    _states: Sequence[FsmState]

    def __init__(self, states: Sequence[FsmState], table: FsmTableWithContext[T_FsmInputs, T_FsmContext]) -> None:
        self._states = states
        self._table = table
        self._state_dict = {type(s): s for s in states}
        self._state = self._state_dict[type(states[0])]

    def _transition(self, inputs: T_FsmInputs, new_state: Type[FsmState[T_FsmInputs, T_FsmContext]], action: Optional[Callable[[T_FsmInputs], None]]) -> None:
        if action:
            action(inputs)
        context = None
        if isinstance(self._state, FsmStateExit):
            context = self._state.on_exit(inputs)
        prev_state = type(self._state)
        if prev_state == new_state:
            if isinstance(self._state, FsmStateStay):
                self._state.on_stay(inputs)
        else:
            self._state = self._state_dict[new_state]
            if context and isinstance(self._state, FsmStateEnterWithContext):
                self._state.on_enter(inputs, context=context)
            elif isinstance(self._state, FsmStateEnter):
                self._state.on_enter(inputs)

    def _check_transitions(self, inputs: T_FsmInputs) -> None:
        for entry in self._table[type(self._state)]:
            if entry.condition(inputs):
                self._transition(inputs, entry.target_state, entry.action)
                return

    def input(self, inputs: T_FsmInputs) -> None:
        if isinstance(self._state, FsmStateCheck):
            self._state.on_check(inputs)
        self._check_transitions(inputs)
        if isinstance(self._state, FsmStateOutput):
            self._state.on_state(inputs)