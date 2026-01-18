from __future__ import annotations
from collections.abc import Callable
from collections import OrderedDict
from typing import Type
import logging
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.providers.provider import ProviderV1
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.providerutils import filter_backends
from .basic_simulator import BasicSimulator

        Return an instance of a backend from its class.

        Args:
            backend_cls (class): backend class.
        Returns:
            Backend: a backend instance.
        Raises:
            QiskitError: if the backend could not be instantiated.
        