import os
import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from statsmodels.genmod.tests.results import glm_test_resids
class Cpunish_offset(Cpunish):
    """
    Same model as Cpunish but with offset of 100.  Many things do not change.
    """

    def __init__(self):
        super().__init__()
        self.params = (-11.40665, 0.0002611017, 0.07781801, -0.09493111, 0.2969349, 2.301183, -18.72207)
        self.bse = (4.147, 5.187e-05, 0.0794, 0.02292, 0.4375, 0.4284, 4.284)