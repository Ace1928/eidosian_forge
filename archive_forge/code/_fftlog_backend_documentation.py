import numpy as np
from warnings import warn
from ._basic import rfft, irfft
from ..special import loggamma, poch
from scipy._lib._array_api import array_namespace, copy
Compute the biased fast Hankel transform.

    This is the basic FFTLog routine.
    