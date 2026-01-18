import logging
import threading
from contextlib import contextmanager
from typing import Dict, Optional, cast, List, Iterator
import httpx
from qcs_api_client.client import QCSClientConfiguration
from qcs_api_client.models.translate_native_quil_to_encrypted_binary_request import (
from qcs_api_client.operations.sync import (
from qcs_api_client.types import UNSET
from rpcq.messages import ParameterSpec
from pyquil.api._abstract_compiler import AbstractCompiler, QuantumExecutable, EncryptedProgram
from pyquil.api._qcs_client import qcs_client
from pyquil.api._rewrite_arithmetic import rewrite_arithmetic
from pyquil.parser import parse_program, parse
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, ExpressionDesignator
from pyquil.quilbase import Declare, Gate
@contextmanager
def _qcs_client(self) -> Iterator[httpx.Client]:
    with qcs_client(client_configuration=self._client_configuration, request_timeout=self._timeout) as client:
        yield client