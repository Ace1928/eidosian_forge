import datetime
from qiskit.providers.models import (
from qiskit.providers.models.backendproperties import Nduv, Gate, BackendProperties
from qiskit.qobj import PulseQobjInstruction
from .fake_backend import FakeBackend
Return the measured characteristics of the backend.