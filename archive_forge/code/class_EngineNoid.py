from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string
class EngineNoid(EngineReturn):
    """A put mock class for testing error responses
    This will not return an id at the first call"""

    def __init__(self):
        self.test_dict = {'status': 'queued'}
        self.counter = 0