from typing import List, cast, Optional, Union, Dict, Any
import functools
from math import sqrt
import httpx
import numpy as np
import networkx as nx
import cirq
from pyquil.quantum_processor import QCSQuantumProcessor
from qcs_api_client.models import InstructionSetArchitecture
from qcs_api_client.operations.sync import get_instruction_set_architecture
from cirq_rigetti._qcs_api_client_decorator import _provide_default_client
Initializes an `AspenQubit` at the given index. See `OctagonalQubit` to understand
        OctagonalQubit indexing.

        Args:
            index: The index at which to initialize the `AspenQubit`.

        Returns:
            The AspenQubit with requested index.

        Raises:
            ValueError: index is not a valid octagon position.
        