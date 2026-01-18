import inspect
import os
import warnings
import param
from packaging.version import Version
class HoloviewsDeprecationWarning(DeprecationWarning):
    """A Holoviews-specific ``DeprecationWarning`` subclass.
    Used to selectively filter Holoviews deprecations for unconditional display.
    """