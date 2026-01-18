from __future__ import annotations
from typing import Any
from collections.abc import Sequence
import numpy as np
from scipy import optimize
from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
def _fit_constrained(self, params):
    """
        TODO: how to add constraints?

        Something like
        sm.add_constraint(Model, func)

        or

        model_instance.add_constraint(func)
        model_instance.add_constraint("x1 + x2 = 2")
        result = model_instance.fit()
        """
    raise NotImplementedError