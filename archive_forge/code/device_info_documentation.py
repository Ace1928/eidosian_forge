from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Union, Optional
from qiskit import pulse
from qiskit.providers import BackendConfigurationError
from qiskit.providers.backend import Backend
Initialize a class with backend information provided by provider.

        Args:
            backend: Backend object.

        Returns:
            OpenPulseBackendInfo: New configured instance.
        