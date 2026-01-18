import inspect
import os
import warnings
import holoviews as hv
import param
from packaging.version import Version
class GeoviewsDeprecationWarning(DeprecationWarning):
    """A Geoviews-specific ``DeprecationWarning`` subclass.
    Used to selectively filter Geoviews deprecations for unconditional display.
    """