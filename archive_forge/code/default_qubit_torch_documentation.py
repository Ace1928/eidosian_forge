import warnings
import inspect
import logging
import semantic_version
import numpy as np
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from . import DefaultQubitLegacy
Sample from the computational basis states based on the state
        probability.

        This is an auxiliary method to the ``generate_samples`` method.

        Args:
            number_of_states (int): the number of basis states to sample from
            state_probability (torch.Tensor[float]): the computational basis probability vector

        Returns:
            List[int]: the sampled basis states
        