from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string
class EngineNoStatus2(EngineReturn):
    """A put mock class for testing error responses
    This will not return a status in the second call"""

    def update(self, *args, **kwargs):
        if self.counter >= 1:
            del self.test_dict['status']
        return self