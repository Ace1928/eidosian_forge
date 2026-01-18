import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
class OdrStop(Exception):
    """
    Exception stopping fitting.

    You can raise this exception in your objective function to tell
    `~scipy.odr.odr` to stop fitting.
    """
    pass