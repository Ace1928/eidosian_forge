from typing import Sequence, Optional, Union, Collection
from cirq import devices, ops, protocols
import numpy as np
Checks that unitary of ControlledGate(gate) is consistent with gate.controlled().