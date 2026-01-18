from typing import cast
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.api.v1 import params_pb2
def _single_param_sweep_to_proto(sweep: sweeps.SingleSweep) -> params_pb2.SingleSweep:
    if isinstance(sweep, cirq.Linspace) and (not isinstance(sweep.key, sympy.Expr)):
        return params_pb2.SingleSweep(parameter_key=sweep.key, linspace=params_pb2.Linspace(first_point=sweep.start, last_point=sweep.stop, num_points=sweep.length))
    elif isinstance(sweep, cirq.Points) and (not isinstance(sweep.key, sympy.Expr)):
        return params_pb2.SingleSweep(parameter_key=sweep.key, points=params_pb2.Points(points=sweep.points))
    else:
        raise ValueError(f'invalid single-parameter sweep: {sweep}')