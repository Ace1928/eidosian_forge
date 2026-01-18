import abc
from typing import Iterable, Sequence, TYPE_CHECKING, List
from cirq import _import, ops, protocols, devices
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
class NoiseModelFromNoiseProperties(devices.NoiseModel):

    def __init__(self, noise_properties: NoiseProperties) -> None:
        """Creates a Noise Model from a NoiseProperties object that can be used with a Simulator.

        Args:
            noise_properties: the NoiseProperties object to be converted to a Noise Model.

        Raises:
            ValueError: if no NoiseProperties object is specified.
        """
        self._noise_properties = noise_properties
        self.noise_models = self._noise_properties.build_noise_models()

    def is_virtual(self, op: 'cirq.Operation') -> bool:
        """Returns True if an operation is virtual.

        Device-specific subclasses should implement this method to mark any
        operations which their device handles outside the quantum hardware.

        Args:
            op: an operation to check for virtual indicators.

        Returns:
            True if `op` is virtual.
        """
        return False

    def noisy_moments(self, moments: Iterable['cirq.Moment'], system_qubits: Sequence['cirq.Qid']) -> Sequence['cirq.OP_TREE']:
        split_measure_moments = []
        multi_measurements = {}
        for moment in moments:
            split_measure_ops = []
            for op in moment:
                if not protocols.is_measurement(op):
                    split_measure_ops.append(op)
                    continue
                m_key = protocols.measurement_key_obj(op)
                multi_measurements[m_key] = op
                for q in op.qubits:
                    split_measure_ops.append(ops.measure(q, key=m_key))
            split_measure_moments.append(circuits.Moment(split_measure_ops))
        new_moments = []
        for moment in split_measure_moments:
            virtual_ops = {op for op in moment if self.is_virtual(op)}
            physical_ops = [op.with_tags(PHYSICAL_GATE_TAG) for op in moment if op not in virtual_ops]
            if virtual_ops:
                new_moments.append(circuits.Moment(virtual_ops))
            if physical_ops:
                new_moments.append(circuits.Moment(physical_ops))
        split_measure_circuit = circuits.Circuit(new_moments)
        noisy_circuit = split_measure_circuit.copy()
        for model in self.noise_models:
            noisy_circuit = noisy_circuit.with_noise(model)
        final_moments = []
        for moment in noisy_circuit:
            combined_measure_ops = []
            restore_keys = set()
            for op in moment:
                if not protocols.is_measurement(op):
                    combined_measure_ops.append(op)
                    continue
                restore_keys.add(protocols.measurement_key_obj(op))
            for key in restore_keys:
                combined_measure_ops.append(multi_measurements[key])
            final_moments.append(circuits.Moment(combined_measure_ops))
        return final_moments