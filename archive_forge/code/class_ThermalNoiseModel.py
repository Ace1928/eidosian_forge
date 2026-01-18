import dataclasses
import functools
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import sympy
from cirq import devices, ops, protocols, qis
from cirq._import import LazyLoader
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
@dataclasses.dataclass
class ThermalNoiseModel(devices.NoiseModel):
    """NoiseModel representing simulated thermalization of a qubit.

    This model is designed for qubits which use energy levels as their states.
    "Heating" and "cooling" here are used to refer to environmental noise which
    transitions a qubit to higher or lower energy levels, respectively.
    """

    def __init__(self, qubits: Set['cirq.Qid'], gate_durations_ns: Dict[type, float], heat_rate_GHz: Union[float, Dict['cirq.Qid', float], None]=None, cool_rate_GHz: Union[float, Dict['cirq.Qid', float], None]=None, dephase_rate_GHz: Union[float, Dict['cirq.Qid', float], None]=None, require_physical_tag: bool=True, skip_measurements: bool=True, prepend: bool=False):
        """Construct a ThermalNoiseModel data object.

        Required Args:
            qubits: Set of all qubits in the system.
            gate_durations_ns: Map of gate types to their duration in
                nanoseconds. These values will override default values for
                gate duration, if any (e.g. WaitGate).
        Optional Args:
            heat_rate_GHz: single number (units GHz) specifying heating rate,
                either per qubit, or global value for all.
                Given a rate gh, the Lindblad op will be sqrt(gh)*a^dag
                (where a is annihilation), so that the heating Lindbladian is
                gh(a^dag • a - 0.5{a*a^dag, •}).
            cool_rate_GHz: single number (units GHz) specifying cooling rate,
                either per qubit, or global value for all.
                Given a rate gc, the Lindblad op will be sqrt(gc)*a
                so that the cooling Lindbladian is gc(a • a^dag - 0.5{n, •})
                This number is equivalent to 1/T1.
            dephase_rate_GHz: single number (units GHz) specifying dephasing
                rate, either per qubit, or global value for all.
                Given a rate gd, Lindblad op will be sqrt(2*gd)*n where
                n = a^dag * a, so that the dephasing Lindbladian is
                2 * gd * (n • n - 0.5{n^2, •}).
                This number is equivalent to 1/Tphi.
            require_physical_tag: whether to only apply noise to operations
                tagged with PHYSICAL_GATE_TAG.
            skip_measurements: whether to skip applying noise to measurements.
            prepend: If True, put noise before affected gates. Default: False.

        Returns:
            The ThermalNoiseModel with specified parameters.
        """
        rate_dict = {}
        heat_rate_GHz = _as_rate_dict(heat_rate_GHz, qubits)
        cool_rate_GHz = _as_rate_dict(cool_rate_GHz, qubits)
        dephase_rate_GHz = _as_rate_dict(dephase_rate_GHz, qubits)
        for q in qubits:
            gamma_h = heat_rate_GHz[q]
            gamma_c = cool_rate_GHz[q]
            gamma_phi = dephase_rate_GHz[q]
            rate_dict[q] = _decoherence_matrix(gamma_c, gamma_phi, gamma_h, q.dimension)
        _validate_rates(qubits, rate_dict)
        self.gate_durations_ns: Dict[type, float] = gate_durations_ns
        self.rate_matrix_GHz: Dict['cirq.Qid', np.ndarray] = rate_dict
        self.require_physical_tag: bool = require_physical_tag
        self.skip_measurements: bool = skip_measurements
        self._prepend = prepend

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        if not moment.operations:
            return [moment]
        if self.require_physical_tag:
            physical_ops = [PHYSICAL_GATE_TAG in op.tags for op in moment]
            if any(physical_ops):
                if not all(physical_ops):
                    raise ValueError(f'Moments are expected to be all physical or all virtual ops, but found {moment.operations}')
            else:
                return [moment]
        noise_ops: List['cirq.Operation'] = []
        moment_ns: float = 0
        for op in moment:
            op_duration: Optional[float] = None
            for key, duration in self.gate_durations_ns.items():
                if not issubclass(type(op.gate), key):
                    continue
                op_duration = duration
                break
            if op_duration is None and isinstance(op.gate, ops.WaitGate):
                nanos = op.gate.duration.total_nanos()
                if isinstance(nanos, sympy.Expr):
                    raise ValueError('Symbolic wait times are not supported')
                op_duration = nanos
            if op_duration is not None:
                moment_ns = max(moment_ns, op_duration)
        if moment_ns == 0:
            return [moment]
        for qubit in system_qubits:
            qubit_op = moment.operation_at(qubit)
            if self.skip_measurements and protocols.is_measurement(qubit_op):
                continue
            rates = self.rate_matrix_GHz[qubit] * moment_ns
            kraus_ops = _kraus_ops_from_rates(tuple(rates.reshape(-1)), rates.shape)
            noise_ops.append(ops.KrausChannel(kraus_ops).on(qubit))
        if not noise_ops:
            return [moment]
        output = [moment, moment_module.Moment(noise_ops)]
        return output[::-1] if self._prepend else output