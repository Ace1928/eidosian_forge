from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Union, Optional
from qiskit import pulse
from qiskit.providers import BackendConfigurationError
from qiskit.providers.backend import Backend
class DrawerBackendInfo(ABC):
    """Backend information to be used for the drawing data generation."""

    def __init__(self, name: Optional[str]=None, dt: Optional[float]=None, channel_frequency_map: Optional[Dict[pulse.channels.Channel, float]]=None, qubit_channel_map: Optional[Dict[int, List[pulse.channels.Channel]]]=None):
        """Create new backend information.

        Args:
            name: Name of the backend.
            dt: System cycle time.
            channel_frequency_map: Mapping of channel and associated frequency.
            qubit_channel_map: Mapping of qubit and associated channels.
        """
        self.backend_name = name or 'no-backend'
        self._dt = dt
        self._chan_freq_map = channel_frequency_map or {}
        self._qubit_channel_map = qubit_channel_map or {}

    @classmethod
    @abstractmethod
    def create_from_backend(cls, backend: Backend):
        """Initialize a class with backend information provided by provider.

        Args:
            backend: Backend object.
        """
        raise NotImplementedError

    @property
    def dt(self):
        """Return cycle time."""
        return self._dt

    def get_qubit_index(self, chan: pulse.channels.Channel) -> Union[int, None]:
        """Get associated qubit index of given channel object."""
        for qind, chans in self._qubit_channel_map.items():
            if chan in chans:
                return qind
        return chan.index

    def get_channel_frequency(self, chan: pulse.channels.Channel) -> Union[float, None]:
        """Get frequency of given channel object."""
        return self._chan_freq_map.get(chan, None)