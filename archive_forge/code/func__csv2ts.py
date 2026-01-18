import numpy as np
import tempfile
from ...utils.misc import package_check
from ...utils.filemanip import fname_presuffix
from .base import NitimeBaseInterface
from ..base import (
def _csv2ts(self):
    """Read data from the in_file and generate a nitime TimeSeries object"""
    from nitime.timeseries import TimeSeries
    data, roi_names = self._read_csv()
    TS = TimeSeries(data=data, sampling_interval=self.inputs.TR, time_unit='s')
    TS.metadata = dict(ROIs=roi_names)
    return TS