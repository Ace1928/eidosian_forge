import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, Type, TypeVar, Union
def _check_transitions(self, inputs: T_FsmInputs) -> None:
    for entry in self._table[type(self._state)]:
        if entry.condition(inputs):
            self._transition(inputs, entry.target_state, entry.action)
            return