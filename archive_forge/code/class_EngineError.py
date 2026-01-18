from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string
class EngineError(EngineReturn):
    """A put mock class for testing error responses"""

    def __init__(self):
        self.test_dict = {'status': 'error', 'id': '2131da', 'samples': 'Error message'}
        self.counter = 0