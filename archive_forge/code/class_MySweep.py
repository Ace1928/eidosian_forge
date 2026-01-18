import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
class MySweep(cirq.study.sweeps.SingleSweep):
    """A sweep that is not serializable."""

    def _tuple(self):
        pass

    def _values(self):
        return ()

    def __len__(self):
        return 0