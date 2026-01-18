from typing import cast
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.api.v1 import params_pb2
def _sweep_from_single_param_sweep_proto(single_param_sweep: params_pb2.SingleSweep) -> cirq.Sweep:
    key = single_param_sweep.parameter_key
    if single_param_sweep.HasField('points'):
        points = single_param_sweep.points
        return cirq.Points(key, list(points.points))
    if single_param_sweep.HasField('linspace'):
        sl = single_param_sweep.linspace
        return cirq.Linspace(key, sl.first_point, sl.last_point, sl.num_points)
    raise ValueError('Single param sweep type undefined')