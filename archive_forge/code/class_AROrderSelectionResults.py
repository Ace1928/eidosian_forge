from __future__ import annotations
from statsmodels.compat.pandas import (
from collections.abc import Iterable
import datetime
import datetime as dt
from types import SimpleNamespace
from typing import Any, Literal, cast
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import eval_measures
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.tsatools import freq_to_period, lagmat
import warnings
class AROrderSelectionResults:
    """
    Results from an AR order selection

    Contains the information criteria for all fitted model orders.
    """

    def __init__(self, model: AutoReg, ics: list[tuple[int | tuple[int, ...], tuple[float, float, float]]], trend: Literal['n', 'c', 'ct', 'ctt'], seasonal: bool, period: int | None):
        self._model = model
        self._ics = ics
        self._trend = trend
        self._seasonal = seasonal
        self._period = period
        aic = sorted(ics, key=lambda r: r[1][0])
        self._aic = {key: val[0] for key, val in aic}
        bic = sorted(ics, key=lambda r: r[1][1])
        self._bic = {key: val[1] for key, val in bic}
        hqic = sorted(ics, key=lambda r: r[1][2])
        self._hqic = {key: val[2] for key, val in hqic}

    @property
    def model(self) -> AutoReg:
        """The model selected using the chosen information criterion."""
        return self._model

    @property
    def seasonal(self) -> bool:
        """Flag indicating if a seasonal component is included."""
        return self._seasonal

    @property
    def trend(self) -> Literal['n', 'c', 'ct', 'ctt']:
        """The trend included in the model selection."""
        return self._trend

    @property
    def period(self) -> int | None:
        """The period of the seasonal component."""
        return self._period

    @property
    def aic(self) -> dict[int | tuple[int, ...], float]:
        """
        The Akaike information criterion for the models fit.

        Returns
        -------
        dict[tuple, float]
        """
        return self._aic

    @property
    def bic(self) -> dict[int | tuple[int, ...], float]:
        """
        The Bayesian (Schwarz) information criteria for the models fit.

        Returns
        -------
        dict[tuple, float]
        """
        return self._bic

    @property
    def hqic(self) -> dict[int | tuple[int, ...], float]:
        """
        The Hannan-Quinn information criteria for the models fit.

        Returns
        -------
        dict[tuple, float]
        """
        return self._hqic

    @property
    def ar_lags(self) -> list[int] | None:
        """The lags included in the selected model."""
        return self._model.ar_lags