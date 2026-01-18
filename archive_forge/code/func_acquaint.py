import functools
import itertools
import math
import operator
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, protocols, value
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
def acquaint(*qubits) -> 'cirq.Operation':
    return AcquaintanceOpportunityGate(len(qubits)).on(*qubits)