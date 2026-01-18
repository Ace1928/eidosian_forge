import abc
from collections.abc import Iterable
from numbers import Number
from typing import Callable, Union, Sequence, Tuple, Optional
from pennylane.measurements import Shots
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.wires import Wires
from pennylane import Tracker
from pennylane.transforms.core import TransformProgram
from .execution_config import ExecutionConfig, DefaultExecutionConfig
Whether or not a given device defines a custom vector jacobian product.

        Args:
            execution_config (ExecutionConfig): A description of the hyperparameters for the desired computation.
            circuit (None, QuantumTape): A specific circuit to check differentation for.

        Default behaviour assumes this to be ``True`` if :meth:`~.compute_vjp` is overridden.
        