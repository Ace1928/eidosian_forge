import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, Type, TypeVar, Union
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