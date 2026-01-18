from copy import copy
from warnings import warn
import pennylane as qml
from pennylane import math
from pennylane.operation import GeneratorUndefinedError
from .exp import Exp
Generator of an operator that is in single-parameter-form.

        For example, for operator

        .. math::

            U(\phi) = e^{-i\phi (0.5 Y + Z\otimes X)}

        we get the generator

        >>> U.generator()
          (0.5) [Y0]
        + (1.0) [Z0 X1]

        