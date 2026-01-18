import re
import copy
import numbers
from typing import Dict, List, Any, Iterable, Tuple, Union
from collections import defaultdict
from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse.channels import (
class PulseBackendConfiguration(QasmBackendConfiguration):
    """Static configuration state for an OpenPulse enabled backend. This contains information
    about the set up of the device which can be useful for building Pulse programs.
    """

    def __init__(self, backend_name: str, backend_version: str, n_qubits: int, basis_gates: List[str], gates: GateConfig, local: bool, simulator: bool, conditional: bool, open_pulse: bool, memory: bool, max_shots: int, coupling_map, n_uchannels: int, u_channel_lo: List[List[UchannelLO]], meas_levels: List[int], qubit_lo_range: List[List[float]], meas_lo_range: List[List[float]], dt: float, dtm: float, rep_times: List[float], meas_kernels: List[str], discriminators: List[str], hamiltonian: Dict[str, Any]=None, channel_bandwidth=None, acquisition_latency=None, conditional_latency=None, meas_map=None, max_experiments=None, sample_name=None, n_registers=None, register_map=None, configurable=None, credits_required=None, online_date=None, display_name=None, description=None, tags=None, channels: Dict[str, Any]=None, **kwargs):
        """
        Initialize a backend configuration that contains all the extra configuration that is made
        available for OpenPulse backends.

        Args:
            backend_name: backend name.
            backend_version: backend version in the form X.Y.Z.
            n_qubits: number of qubits.
            basis_gates: list of basis gates names on the backend.
            gates: list of basis gates on the backend.
            local: backend is local or remote.
            simulator: backend is a simulator.
            conditional: backend supports conditional operations.
            open_pulse: backend supports open pulse.
            memory: backend supports memory.
            max_shots: maximum number of shots supported.
            coupling_map (list): The coupling map for the device
            n_uchannels: Number of u-channels.
            u_channel_lo: U-channel relationship on device los.
            meas_levels: Supported measurement levels.
            qubit_lo_range: Qubit lo ranges for each qubit with form (min, max) in GHz.
            meas_lo_range: Measurement lo ranges for each qubit with form (min, max) in GHz.
            dt: Qubit drive channel timestep in nanoseconds.
            dtm: Measurement drive channel timestep in nanoseconds.
            rep_times: Supported repetition times (program execution time) for backend in μs.
            meas_kernels: Supported measurement kernels.
            discriminators: Supported discriminators.
            hamiltonian: An optional dictionary with fields characterizing the system hamiltonian.
            channel_bandwidth (list): Bandwidth of all channels
                (qubit, measurement, and U)
            acquisition_latency (list): Array of dimension
                n_qubits x n_registers. Latency (in units of dt) to write a
                measurement result from qubit n into register slot m.
            conditional_latency (list): Array of dimension n_channels
                [d->u->m] x n_registers. Latency (in units of dt) to do a
                conditional operation on channel n from register slot m
            meas_map (list): Grouping of measurement which are multiplexed
            max_experiments (int): The maximum number of experiments per job
            sample_name (str): Sample name for the backend
            n_registers (int): Number of register slots available for feedback
                (if conditional is True)
            register_map (list): An array of dimension n_qubits X
                n_registers that specifies whether a qubit can store a
                measurement in a certain register slot.
            configurable (bool): True if the backend is configurable, if the
                backend is a simulator
            credits_required (bool): True if backend requires credits to run a
                job.
            online_date (datetime.datetime): The date that the device went online
            display_name (str): Alternate name field for the backend
            description (str): A description for the backend
            tags (list): A list of string tags to describe the backend
            channels: An optional dictionary containing information of each channel -- their
                purpose, type, and qubits operated on.
            **kwargs: Optional fields.
        """
        self.n_uchannels = n_uchannels
        self.u_channel_lo = u_channel_lo
        self.meas_levels = meas_levels
        self.qubit_lo_range = [[min_range * 1000000000.0, max_range * 1000000000.0] for min_range, max_range in qubit_lo_range]
        self.meas_lo_range = [[min_range * 1000000000.0, max_range * 1000000000.0] for min_range, max_range in meas_lo_range]
        self.meas_kernels = meas_kernels
        self.discriminators = discriminators
        self.hamiltonian = hamiltonian
        if hamiltonian is not None:
            self.hamiltonian = dict(hamiltonian)
            self.hamiltonian['vars'] = {k: v * 1000000000.0 if isinstance(v, numbers.Number) else v for k, v in self.hamiltonian['vars'].items()}
        self.rep_times = [_rt * 1e-06 for _rt in rep_times]
        self.dt = dt * 1e-09
        self.dtm = dtm * 1e-09
        if channels is not None:
            self.channels = channels
            self._qubit_channel_map, self._channel_qubit_map, self._control_channels = self._parse_channels(channels=channels)
        else:
            self._control_channels = defaultdict(list)
        if channel_bandwidth is not None:
            self.channel_bandwidth = [[min_range * 1000000000.0, max_range * 1000000000.0] for min_range, max_range in channel_bandwidth]
        if acquisition_latency is not None:
            self.acquisition_latency = acquisition_latency
        if conditional_latency is not None:
            self.conditional_latency = conditional_latency
        if meas_map is not None:
            self.meas_map = meas_map
        super().__init__(backend_name=backend_name, backend_version=backend_version, n_qubits=n_qubits, basis_gates=basis_gates, gates=gates, local=local, simulator=simulator, conditional=conditional, open_pulse=open_pulse, memory=memory, max_shots=max_shots, coupling_map=coupling_map, max_experiments=max_experiments, sample_name=sample_name, n_registers=n_registers, register_map=register_map, configurable=configurable, credits_required=credits_required, online_date=online_date, display_name=display_name, description=description, tags=tags, **kwargs)

    @classmethod
    def from_dict(cls, data):
        """Create a new GateConfig object from a dictionary.

        Args:
            data (dict): A dictionary representing the GateConfig to create.
                It will be in the same format as output by :func:`to_dict`.

        Returns:
            GateConfig: The GateConfig from the input dictionary.
        """
        in_data = copy.copy(data)
        gates = [GateConfig.from_dict(x) for x in in_data.pop('gates')]
        in_data['gates'] = gates
        input_uchannels = in_data.pop('u_channel_lo')
        u_channels = []
        for channel in input_uchannels:
            u_channels.append([UchannelLO.from_dict(x) for x in channel])
        in_data['u_channel_lo'] = u_channels
        return cls(**in_data)

    def to_dict(self):
        """Return a dictionary format representation of the GateConfig.

        Returns:
            dict: The dictionary form of the GateConfig.
        """
        out_dict = super().to_dict()
        u_channel_lo = []
        for x in self.u_channel_lo:
            channel = []
            for y in x:
                channel.append(y.to_dict())
            u_channel_lo.append(channel)
        out_dict.update({'n_uchannels': self.n_uchannels, 'u_channel_lo': u_channel_lo, 'meas_levels': self.meas_levels, 'qubit_lo_range': self.qubit_lo_range, 'meas_lo_range': self.meas_lo_range, 'meas_kernels': self.meas_kernels, 'discriminators': self.discriminators, 'rep_times': self.rep_times, 'dt': self.dt, 'dtm': self.dtm})
        if hasattr(self, 'channel_bandwidth'):
            out_dict['channel_bandwidth'] = self.channel_bandwidth
        if hasattr(self, 'meas_map'):
            out_dict['meas_map'] = self.meas_map
        if hasattr(self, 'acquisition_latency'):
            out_dict['acquisition_latency'] = self.acquisition_latency
        if hasattr(self, 'conditional_latency'):
            out_dict['conditional_latency'] = self.conditional_latency
        if 'channels' in out_dict:
            out_dict.pop('_qubit_channel_map')
            out_dict.pop('_channel_qubit_map')
            out_dict.pop('_control_channels')
        if self.qubit_lo_range:
            out_dict['qubit_lo_range'] = [[min_range * 1e-09, max_range * 1e-09] for min_range, max_range in self.qubit_lo_range]
        if self.meas_lo_range:
            out_dict['meas_lo_range'] = [[min_range * 1e-09, max_range * 1e-09] for min_range, max_range in self.meas_lo_range]
        if self.rep_times:
            out_dict['rep_times'] = [_rt * 1000000.0 for _rt in self.rep_times]
        out_dict['dt'] *= 1000000000.0
        out_dict['dtm'] *= 1000000000.0
        if hasattr(self, 'channel_bandwidth'):
            out_dict['channel_bandwidth'] = [[min_range * 1e-09, max_range * 1e-09] for min_range, max_range in self.channel_bandwidth]
        if self.hamiltonian:
            hamiltonian = copy.deepcopy(self.hamiltonian)
            hamiltonian['vars'] = {k: v * 1e-09 if isinstance(v, numbers.Number) else v for k, v in hamiltonian['vars'].items()}
            out_dict['hamiltonian'] = hamiltonian
        if hasattr(self, 'channels'):
            out_dict['channels'] = self.channels
        return out_dict

    def __eq__(self, other):
        if isinstance(other, QasmBackendConfiguration):
            if self.to_dict() == other.to_dict():
                return True
        return False

    @property
    def sample_rate(self) -> float:
        """Sample rate of the signal channels in Hz (1/dt)."""
        return 1.0 / self.dt

    @property
    def control_channels(self) -> Dict[Tuple[int, ...], List]:
        """Return the control channels"""
        return self._control_channels

    def drive(self, qubit: int) -> DriveChannel:
        """
        Return the drive channel for the given qubit.

        Raises:
            BackendConfigurationError: If the qubit is not a part of the system.

        Returns:
            Qubit drive channel.
        """
        if not 0 <= qubit < self.n_qubits:
            raise BackendConfigurationError(f'Invalid index for {qubit}-qubit system.')
        return DriveChannel(qubit)

    def measure(self, qubit: int) -> MeasureChannel:
        """
        Return the measure stimulus channel for the given qubit.

        Raises:
            BackendConfigurationError: If the qubit is not a part of the system.
        Returns:
            Qubit measurement stimulus line.
        """
        if not 0 <= qubit < self.n_qubits:
            raise BackendConfigurationError(f'Invalid index for {qubit}-qubit system.')
        return MeasureChannel(qubit)

    def acquire(self, qubit: int) -> AcquireChannel:
        """
        Return the acquisition channel for the given qubit.

        Raises:
            BackendConfigurationError: If the qubit is not a part of the system.
        Returns:
            Qubit measurement acquisition line.
        """
        if not 0 <= qubit < self.n_qubits:
            raise BackendConfigurationError(f'Invalid index for {qubit}-qubit systems.')
        return AcquireChannel(qubit)

    def control(self, qubits: Iterable[int]=None) -> List[ControlChannel]:
        """
        Return the secondary drive channel for the given qubit -- typically utilized for
        controlling multiqubit interactions. This channel is derived from other channels.

        Args:
            qubits: Tuple or list of qubits of the form `(control_qubit, target_qubit)`.

        Raises:
            BackendConfigurationError: If the ``qubits`` is not a part of the system or if
                the backend does not provide `channels` information in its configuration.

        Returns:
            List of control channels.
        """
        try:
            if isinstance(qubits, list):
                qubits = tuple(qubits)
            return self._control_channels[qubits]
        except KeyError as ex:
            raise BackendConfigurationError(f"Couldn't find the ControlChannel operating on qubits {qubits} on {self.n_qubits}-qubit system. The ControlChannel information is retrieved from the backend.") from ex
        except AttributeError as ex:
            raise BackendConfigurationError(f"This backend - '{self.backend_name}' does not provide channel information.") from ex

    def get_channel_qubits(self, channel: Channel) -> List[int]:
        """
        Return a list of indices for qubits which are operated on directly by the given ``channel``.

        Raises:
            BackendConfigurationError: If ``channel`` is not a found or if
                the backend does not provide `channels` information in its configuration.

        Returns:
            List of qubits operated on my the given ``channel``.
        """
        try:
            return self._channel_qubit_map[channel]
        except KeyError as ex:
            raise BackendConfigurationError(f"Couldn't find the Channel - {channel}") from ex
        except AttributeError as ex:
            raise BackendConfigurationError(f"This backend - '{self.backend_name}' does not provide channel information.") from ex

    def get_qubit_channels(self, qubit: Union[int, Iterable[int]]) -> List[Channel]:
        """Return a list of channels which operate on the given ``qubit``.

        Raises:
            BackendConfigurationError: If ``qubit`` is not a found or if
                the backend does not provide `channels` information in its configuration.

        Returns:
            List of ``Channel``\\s operated on my the given ``qubit``.
        """
        channels = set()
        try:
            if isinstance(qubit, int):
                for key in self._qubit_channel_map.keys():
                    if qubit in key:
                        channels.update(self._qubit_channel_map[key])
                if len(channels) == 0:
                    raise KeyError
            elif isinstance(qubit, list):
                qubit = tuple(qubit)
                channels.update(self._qubit_channel_map[qubit])
            elif isinstance(qubit, tuple):
                channels.update(self._qubit_channel_map[qubit])
            return list(channels)
        except KeyError as ex:
            raise BackendConfigurationError(f"Couldn't find the qubit - {qubit}") from ex
        except AttributeError as ex:
            raise BackendConfigurationError(f"This backend - '{self.backend_name}' does not provide channel information.") from ex

    def describe(self, channel: ControlChannel) -> Dict[DriveChannel, complex]:
        """
        Return a basic description of the channel dependency. Derived channels are given weights
        which describe how their frames are linked to other frames.
        For instance, the backend could be configured with this setting::

            u_channel_lo = [
                [UchannelLO(q=0, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=-1. + 0.j), UchannelLO(q=1, scale=1. + 0.j)]
            ]

        Then, this method can be used as follows::

            backend.configuration().describe(ControlChannel(1))
            >>> {DriveChannel(0): -1, DriveChannel(1): 1}

        Args:
            channel: The derived channel to describe.
        Raises:
            BackendConfigurationError: If channel is not a ControlChannel.
        Returns:
            Control channel derivations.
        """
        if not isinstance(channel, ControlChannel):
            raise BackendConfigurationError('Can only describe ControlChannels.')
        result = {}
        for u_chan_lo in self.u_channel_lo[channel.index]:
            result[DriveChannel(u_chan_lo.q)] = u_chan_lo.scale
        return result

    def _parse_channels(self, channels: Dict[set, Any]) -> Dict[Any, Any]:
        """
        Generates a dictionaries of ``Channel``\\s, and tuple of qubit(s) they operate on.

        Args:
            channels: An optional dictionary containing information of each channel -- their
                purpose, type, and qubits operated on.

        Returns:
            qubit_channel_map: Dictionary mapping tuple of qubit(s) to list of ``Channel``\\s.
            channel_qubit_map: Dictionary mapping ``Channel`` to list of qubit(s).
            control_channels: Dictionary mapping tuple of qubit(s), to list of
                ``ControlChannel``\\s.
        """
        qubit_channel_map = defaultdict(list)
        channel_qubit_map = defaultdict(list)
        control_channels = defaultdict(list)
        channels_dict = {DriveChannel.prefix: DriveChannel, ControlChannel.prefix: ControlChannel, MeasureChannel.prefix: MeasureChannel, 'acquire': AcquireChannel}
        for channel, config in channels.items():
            channel_prefix, index = self._get_channel_prefix_index(channel)
            channel_type = channels_dict[channel_prefix]
            qubits = tuple(config['operates']['qubits'])
            if channel_prefix in channels_dict:
                qubit_channel_map[qubits].append(channel_type(index))
                channel_qubit_map[channel_type(index)].extend(list(qubits))
                if channel_prefix == ControlChannel.prefix:
                    control_channels[qubits].append(channel_type(index))
        return (dict(qubit_channel_map), dict(channel_qubit_map), dict(control_channels))

    def _get_channel_prefix_index(self, channel: str) -> str:
        """Return channel prefix and index from the given ``channel``.

        Args:
            channel: Name of channel.

        Raises:
            BackendConfigurationError: If invalid channel name is found.

        Return:
            Channel name and index. For example, if ``channel=acquire0``, this method
            returns ``acquire`` and ``0``.
        """
        channel_prefix = re.match('(?P<channel>[a-z]+)(?P<index>[0-9]+)', channel)
        try:
            return (channel_prefix.group('channel'), int(channel_prefix.group('index')))
        except AttributeError as ex:
            raise BackendConfigurationError(f"Invalid channel name - '{channel}' found.") from ex