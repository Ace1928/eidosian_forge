import abc
from typing import Generic, Dict, Any, List, Sequence, Union
from unittest import mock
import duet
import numpy as np
import pytest
import cirq
from cirq import study
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulator import (
class FakeMissingIterSimulatorImpl(SimulatesAmplitudes, SimulatesExpectationValues, SimulatesFinalState):
    """A class which fails to define simulator methods."""