from qcs_api_client.models import InstructionSetArchitecture, Characteristic, Operation
from pyquil.external.rpcq import CompilerISA, add_edge, add_qubit, get_qubit, get_edge
import numpy as np
from pyquil.external.rpcq import (
from typing import List, Union, cast, DefaultDict, Set, Optional
from collections import defaultdict
def _make_wildcard_2q_gates() -> List[GateInfo]:
    return [GateInfo(operator='_', parameters=['_'], arguments=['_', '_'], fidelity=PERFECT_FIDELITY, duration=PERFECT_DURATION)]