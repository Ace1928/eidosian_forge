import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def _get_program_gates(prog: 'Program') -> List[Gate]:
    """
    Get all gate applications appearing in prog.

    :param prog: The program
    :return: A list of all Gates in prog (without duplicates).
    """
    return sorted({i for i in prog if isinstance(i, Gate)}, key=lambda g: g.out())