import abc
import dataclasses
from typing import Any, Dict
import cirq
import cirq_google as cg
from cirq._compat import dataclass_repr
from cirq_google.engine.virtual_engine_factory import (
def _get_input_sampler(self) -> 'cirq.Sampler':
    """Return a local `cirq.Sampler` based on the `noise_strength` attribute.

        If `self.noise_strength` is `0` return a noiseless state-vector simulator.
        If it's set to `float('inf')` the simulator will be `cirq.ZerosSampler`.
        Otherwise, we return a density matrix simulator with a depolarizing model with
        `noise_strength` probability of noise.

        This function is used to initialize this class's `processor`.
        """
    if self.noise_strength == 0:
        return cirq.Simulator()
    if self.noise_strength == float('inf'):
        return cirq.ZerosSampler()
    return cirq.DensityMatrixSimulator(noise=cirq.ConstantQubitNoiseModel(cirq.depolarize(p=self.noise_strength)))