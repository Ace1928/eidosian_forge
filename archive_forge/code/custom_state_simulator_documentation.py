from typing import Any, Dict, Generic, Sequence, Type, TYPE_CHECKING
import numpy as np
from cirq import sim
from cirq.sim.simulation_state import TSimulationState
Initializes a CustomStateSimulator.

        Args:
            state_type: The class that represents the simulation state this simulator should use.
            noise: The noise model used by the simulator.
            split_untangled_states: True to run the simulation as a product state. This is only
                supported if the `state_type` supports it via an implementation of `kron` and
                `factor` methods. Otherwise a runtime error will occur during simulation.