import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
class OdrError(Exception):
    """
    Exception indicating an error in fitting.

    This is raised by `~scipy.odr.odr` if an error occurs during fitting.
    """
    pass