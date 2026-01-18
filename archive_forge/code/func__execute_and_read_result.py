from typing import Any, cast, Dict, Optional, Sequence, Union
from pyquil import Program
from pyquil.api import QuantumComputer, QuantumExecutable
from pyquil.quilbase import Declare
import cirq
import sympy
from typing_extensions import Protocol
from cirq_rigetti.logging import logger
from cirq_rigetti import circuit_transformers as transformers
def _execute_and_read_result(quantum_computer: QuantumComputer, executable: QuantumExecutable, measurement_id_map: Dict[str, str], resolver: cirq.ParamResolverOrSimilarType, memory_map: Optional[Dict[Union[sympy.Expr, str], Union[int, float, Sequence[int], Sequence[float]]]]=None) -> cirq.Result:
    """Execute the `pyquil.api.QuantumExecutable` and parse the measurements into
    a `cirq.Result`.

    Args:
        quantum_computer: The `pyquil.api.QuantumComputer` on which to execute
            and from which to read results.
        executable: The fully compiled `pyquil.api.QuantumExecutable` to run.
        measurement_id_map: A dict mapping cirq measurement keys to pyQuil
            read out regions.
        resolver: The `cirq.ParamResolverOrSimilarType` to include on
            the returned `cirq.Result`.
        memory_map: A dict of values to write to memory values on the
            `quantum_computer`. The `pyquil.api.QuantumAbstractMachine` reads these
            v_execute_and_read_resultalues into memory regions on the pre-compiled
            `executable` during execution.

    Returns:
        A `cirq.Result` with measurements read from the `quantum_computer`.

    Raises:
        ValueError: measurement_id_map references an undefined pyQuil readout region.
    """
    if memory_map is None:
        memory_map = {}
    for region_name, values in memory_map.items():
        if isinstance(region_name, str):
            executable.write_memory(region_name=region_name, value=values)
        else:
            raise ValueError(f'Symbols not valid for region name {region_name}')
    qam_execution_result = quantum_computer.qam.run(executable)
    measurements = {}
    for cirq_memory_key, pyquil_region in measurement_id_map.items():
        readout = qam_execution_result.readout_data.get(pyquil_region)
        if readout is None:
            raise ValueError(f'readout data does not have values for region "{pyquil_region}"')
        measurements[cirq_memory_key] = readout
    logger.debug(f'measurement_id_map {measurement_id_map}')
    logger.debug(f'measurements {measurements}')
    result = cirq.ResultDict(params=cast(cirq.ParamResolver, resolver or cirq.ParamResolver({})), measurements=measurements)
    return result