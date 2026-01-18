import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, Type, TypeVar, Union
Finite state machine.

Simple FSM implementation.

Usage:
    ```python
    class A:
        def on_output(self, inputs) -> None:
            pass


    class B:
        def on_output(self, inputs) -> None:
            pass


    def to_b(inputs) -> bool:
        return True


    def to_a(inputs) -> bool:
        return True


    f = Fsm(states=[A(), B()], table={A: [(to_b, B)], B: [(to_a, A)]})
    f.run({"input1": 1, "input2": 2})
    ```
