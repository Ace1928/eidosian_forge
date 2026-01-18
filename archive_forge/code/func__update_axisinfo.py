import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
def _update_axisinfo(self):
    """
        Check the axis converter for the stored units to see if the
        axis info needs to be updated.
        """
    if self.converter is None:
        return
    info = self.converter.axisinfo(self.units, self)
    if info is None:
        return
    if info.majloc is not None and self.major.locator != info.majloc and self.isDefault_majloc:
        self.set_major_locator(info.majloc)
        self.isDefault_majloc = True
    if info.minloc is not None and self.minor.locator != info.minloc and self.isDefault_minloc:
        self.set_minor_locator(info.minloc)
        self.isDefault_minloc = True
    if info.majfmt is not None and self.major.formatter != info.majfmt and self.isDefault_majfmt:
        self.set_major_formatter(info.majfmt)
        self.isDefault_majfmt = True
    if info.minfmt is not None and self.minor.formatter != info.minfmt and self.isDefault_minfmt:
        self.set_minor_formatter(info.minfmt)
        self.isDefault_minfmt = True
    if info.label is not None and self.isDefault_label:
        self.set_label_text(info.label)
        self.isDefault_label = True
    self.set_default_intervals()