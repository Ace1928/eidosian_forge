import inspect
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
from matplotlib.transforms import Transform, IdentityTransform
class ScaleBase:
    """
    The base class for all scales.

    Scales are separable transformations, working on a single dimension.

    Subclasses should override

    :attr:`name`
        The scale's name.
    :meth:`get_transform`
        A method returning a `.Transform`, which converts data coordinates to
        scaled coordinates.  This transform should be invertible, so that e.g.
        mouse positions can be converted back to data coordinates.
    :meth:`set_default_locators_and_formatters`
        A method that sets default locators and formatters for an `~.axis.Axis`
        that uses this scale.
    :meth:`limit_range_for_scale`
        An optional method that "fixes" the axis range to acceptable values,
        e.g. restricting log-scaled axes to positive values.
    """

    def __init__(self, axis):
        """
        Construct a new scale.

        Notes
        -----
        The following note is for scale implementors.

        For back-compatibility reasons, scales take an `~matplotlib.axis.Axis`
        object as first argument.  However, this argument should not
        be used: a single scale object should be usable by multiple
        `~matplotlib.axis.Axis`\\es at the same time.
        """

    def get_transform(self):
        """
        Return the `.Transform` object associated with this scale.
        """
        raise NotImplementedError()

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters of *axis* to instances suitable for
        this scale.
        """
        raise NotImplementedError()

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Return the range *vmin*, *vmax*, restricted to the
        domain supported by this scale (if any).

        *minpos* should be the minimum positive value in the data.
        This is used by log scales to determine a minimum value.
        """
        return (vmin, vmax)