from typing import cast
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.api.v1 import params_pb2
def _sweep_zip_to_proto(sweep: cirq.Zip) -> params_pb2.ZipSweep:
    sweep_list = [_single_param_sweep_to_proto(cast(sweeps.SingleSweep, s)) for s in sweep.sweeps]
    return params_pb2.ZipSweep(sweeps=sweep_list)