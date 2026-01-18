from typing import List, Union, Type, cast, TYPE_CHECKING
from enum import Enum
import numpy as np
from cirq import ops, transformers, protocols, linalg
from cirq.type_workarounds import NotImplementedType
List of transformers which should be run after decomposing individual operations.