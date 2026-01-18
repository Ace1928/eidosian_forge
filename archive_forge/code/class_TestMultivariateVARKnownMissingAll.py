import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
class TestMultivariateVARKnownMissingAll(MultivariateVARKnown):
    """
    Notes
    -----
    Cannot test against KFAS because they have a different behavior for
    missing entries. When an entry is missing, KFAS does not draw a simulation
    smoothed value for that entry, whereas we draw from the unconditional
    distribution. It appears there is nothing to definitively recommend one
    approach over the other, but it makes it difficult to line up the variates
    correctly in order to replicate results.
    """

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(missing='all', test_against_KFAS=False)
        cls.true_llf = 1305.739288