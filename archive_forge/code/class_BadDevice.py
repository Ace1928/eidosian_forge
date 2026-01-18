import itertools
import pytest
import cirq
import cirq_google as cg
import numpy as np
class BadDevice(cirq.Device):
    pass