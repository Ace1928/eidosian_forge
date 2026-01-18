import re
import copy
import numbers
from typing import Dict, List, Any, Iterable, Tuple, Union
from collections import defaultdict
from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse.channels import (
class BackendConfiguration(QasmBackendConfiguration):
    """Backwards compat shim representing an abstract backend configuration."""
    pass