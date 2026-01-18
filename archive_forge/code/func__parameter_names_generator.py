import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
def _parameter_names_generator(self) -> Iterator[str]:
    yield from protocols.parameter_names(self.repetitions)
    for symbol in protocols.parameter_symbols(self.circuit):
        for name in protocols.parameter_names(protocols.resolve_parameters(symbol, self.param_resolver, recursive=False)):
            yield name