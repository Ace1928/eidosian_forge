from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union, Tuple
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil._version import pyquil_version
from pyquil.api import QuantumExecutable
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qvm_client import (
from pyquil.noise import NoiseModel, apply_noise_model
from pyquil.quil import Program, get_classical_addresses_from_program
class QVMNotRunning(Exception):
    pass