import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union
from qiskit import qobj, pulse
from qiskit.assembler.run_config import RunConfig
from qiskit.exceptions import QiskitError
from qiskit.pulse import instructions, transforms, library, schedule, channels
from qiskit.qobj import utils as qobj_utils, converters
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes
def _assemble_config(lo_converter: converters.LoConfigConverter, experiment_config: Dict[str, Any], run_config: RunConfig) -> qobj.PulseQobjConfig:
    """Assembles the QobjConfiguration from experimental config and runtime config.

    Args:
        lo_converter: The configured frequency converter and validator.
        experiment_config: Schedules to assemble.
        run_config: Configuration of the runtime environment.

    Returns:
        The assembled PulseQobjConfig.
    """
    qobj_config = run_config.to_dict()
    qobj_config.update(experiment_config)
    qobj_config.pop('meas_map', None)
    qobj_config.pop('qubit_lo_range', None)
    qobj_config.pop('meas_lo_range', None)
    meas_return = qobj_config.get('meas_return', 'avg')
    if isinstance(meas_return, qobj_utils.MeasReturnType):
        qobj_config['meas_return'] = meas_return.value
    meas_level = qobj_config.get('meas_level', 2)
    if isinstance(meas_level, qobj_utils.MeasLevel):
        qobj_config['meas_level'] = meas_level.value
    qobj_config['qubit_lo_freq'] = [freq / 1000000000.0 for freq in qobj_config['qubit_lo_freq']]
    qobj_config['meas_lo_freq'] = [freq / 1000000000.0 for freq in qobj_config['meas_lo_freq']]
    schedule_los = qobj_config.pop('schedule_los', [])
    if len(schedule_los) == 1:
        lo_dict = schedule_los[0]
        q_los = lo_converter.get_qubit_los(lo_dict)
        if q_los:
            qobj_config['qubit_lo_freq'] = [freq / 1000000000.0 for freq in q_los]
        m_los = lo_converter.get_meas_los(lo_dict)
        if m_los:
            qobj_config['meas_lo_freq'] = [freq / 1000000000.0 for freq in m_los]
    return qobj.PulseQobjConfig(**qobj_config)