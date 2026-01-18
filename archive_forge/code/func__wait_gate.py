import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _wait_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    return 'WAIT\n'