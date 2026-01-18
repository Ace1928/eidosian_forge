from statsmodels.compat.pandas import assert_series_equal
from io import StringIO
import warnings
import numpy as np
import numpy.testing as npt
import pandas as pd
import patsy
import pytest
from statsmodels.datasets import cpunish
from statsmodels.datasets.longley import load, load_pandas
from statsmodels.formula.api import ols
from statsmodels.formula.formulatools import make_hypotheses_matrices
from statsmodels.tools import add_constant
from statsmodels.tools.testing import assert_equal
class TestFormulaPandas(CheckFormulaOLS):

    @classmethod
    def setup_class(cls):
        data = load_pandas().data
        cls.model = ols(longley_formula, data)
        super().setup_class()