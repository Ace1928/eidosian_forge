from typing import Sequence, Callable
from functools import partial
from pennylane import transform
from pennylane.tape import QuantumTape
from .utils import get_spectrum, join_spectra
Process the tapes extract the spectrum of the circuit.