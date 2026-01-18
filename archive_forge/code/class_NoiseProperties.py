import abc
from typing import Iterable, Sequence, TYPE_CHECKING, List
from cirq import _import, ops, protocols, devices
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
class NoiseProperties(abc.ABC):
    """Noise-defining properties for a quantum device."""

    @abc.abstractmethod
    def build_noise_models(self) -> List['cirq.NoiseModel']:
        """Construct all NoiseModels associated with this NoiseProperties."""