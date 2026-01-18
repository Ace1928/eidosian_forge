import copy
import functools
from warnings import warn
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Tuple, Optional, Union
import pennylane as qml
from pennylane.operation import Operator, DecompositionUndefinedError, EigvalsUndefinedError
from pennylane.pytrees import register_pytree
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .shots import Shots
from a classical shadow measurement"""
class ObservableReturnTypes(Enum):
    """Enumeration class to represent the return types of an observable."""
    Sample = 'sample'
    Counts = 'counts'
    AllCounts = 'allcounts'
    Variance = 'var'
    Expectation = 'expval'
    Probability = 'probs'
    State = 'state'
    MidMeasure = 'measure'
    VnEntropy = 'vnentropy'
    MutualInfo = 'mutualinfo'
    Shadow = 'shadow'
    ShadowExpval = 'shadowexpval'
    Purity = 'purity'

    def __repr__(self):
        """String representation of the return types."""
        return str(self.value)