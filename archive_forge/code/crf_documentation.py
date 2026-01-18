import numpy as np
from ..processors import Processor

        Determine the most probable configuration of Y given the state
        sequence x:

        .. math::
            y^* = argmax_y P(Y=y|X=x)

        Parameters
        ----------
        observations : numpy array
            Observations (x) to decode the most probable state sequence for.

        Returns
        -------
        y_star : numpy array
            Most probable state sequence.
        