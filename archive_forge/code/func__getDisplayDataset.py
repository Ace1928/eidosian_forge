import math
import warnings
import bisect
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem
def _getDisplayDataset(self):
    """
        Returns a :class:`~.PlotDataset` object that contains data suitable for display 
        (after mapping and data reduction) as ``dataset.x`` and ``dataset.y``.
        Intended for internal use.
        """
    if self._dataset is None:
        return None
    if self._datasetDisplay is not None and (not (self.property('xViewRangeWasChanged') and self.opts['clipToView'])) and (not (self.property('xViewRangeWasChanged') and self.opts['autoDownsample'])) and (not (self.property('yViewRangeWasChanged') and self.opts['dynamicRangeLimit'] is not None)):
        return self._datasetDisplay
    if self._datasetMapped is None:
        x = self._dataset.x
        y = self._dataset.y
        if y.dtype == bool:
            y = y.astype(np.uint8)
        if x.dtype == bool:
            x = x.astype(np.uint8)
        if self.opts['fftMode']:
            x, y = self._fourierTransform(x, y)
            if self.opts['logMode'][0]:
                x = x[1:]
                y = y[1:]
        if self.opts['derivativeMode']:
            y = np.diff(self._dataset.y) / np.diff(self._dataset.x)
            x = x[:-1]
        if self.opts['phasemapMode']:
            x = self._dataset.y[:-1]
            y = np.diff(self._dataset.y) / np.diff(self._dataset.x)
        dataset = PlotDataset(x, y, self._dataset.xAllFinite, self._dataset.yAllFinite)
        if True in self.opts['logMode']:
            dataset.applyLogMapping(self.opts['logMode'])
        self._datasetMapped = dataset
    x = self._datasetMapped.x
    y = self._datasetMapped.y
    xAllFinite = self._datasetMapped.xAllFinite
    yAllFinite = self._datasetMapped.yAllFinite
    view = self.getViewBox()
    if view is None:
        view_range = None
    else:
        view_range = view.viewRect()
    if view_range is None:
        view_range = self.viewRect()
    ds = self.opts['downsample']
    if not isinstance(ds, int):
        ds = 1
    if self.opts['autoDownsample']:
        if xAllFinite:
            finite_x = x
        else:
            finite_x = x[np.isfinite(x)]
        if view_range is not None and len(finite_x) > 1:
            dx = float(finite_x[-1] - finite_x[0]) / (len(finite_x) - 1)
            if dx != 0.0:
                width = self.getViewBox().width()
                if width != 0.0:
                    ds_float = max(1.0, abs(view_range.width() / dx / (width * self.opts['autoDownsampleFactor'])))
                    if math.isfinite(ds_float):
                        ds = int(ds_float)
        if math.isclose(ds, self._adsLastValue, rel_tol=0.01):
            ds = self._adsLastValue
        self._adsLastValue = ds
    if self.opts['clipToView']:
        if view is None or view.autoRangeEnabled()[0]:
            pass
        elif view_range is not None and len(x) > 1:
            x0 = bisect.bisect_left(x, view_range.left()) - ds
            x0 = fn.clip_scalar(x0, 0, len(x))
            x1 = bisect.bisect_left(x, view_range.right()) + ds
            x1 = fn.clip_scalar(x1, x0, len(x))
            x = x[x0:x1]
            y = y[x0:x1]
    if ds > 1:
        if self.opts['downsampleMethod'] == 'subsample':
            x = x[::ds]
            y = y[::ds]
        elif self.opts['downsampleMethod'] == 'mean':
            n = len(x) // ds
            stx = ds // 2
            x = x[stx:stx + n * ds:ds]
            y = y[:n * ds].reshape(n, ds).mean(axis=1)
        elif self.opts['downsampleMethod'] == 'peak':
            n = len(x) // ds
            x1 = np.empty((n, 2))
            stx = ds // 2
            x1[:] = x[stx:stx + n * ds:ds, np.newaxis]
            x = x1.reshape(n * 2)
            y1 = np.empty((n, 2))
            y2 = y[:n * ds].reshape((n, ds))
            y1[:, 0] = y2.max(axis=1)
            y1[:, 1] = y2.min(axis=1)
            y = y1.reshape(n * 2)
    if self.opts['dynamicRangeLimit'] is not None:
        if view_range is not None:
            data_range = self._datasetMapped.dataRect()
            if data_range is not None:
                view_height = view_range.height()
                limit = self.opts['dynamicRangeLimit']
                hyst = self.opts['dynamicRangeHyst']
                if view_height > 0 and (not data_range.bottom() < view_range.top()) and (not data_range.top() > view_range.bottom()) and (data_range.height() > 2 * hyst * limit * view_height):
                    cache_is_good = False
                    if self._datasetDisplay is not None:
                        top_exc = -(self._drlLastClip[0] - view_range.bottom()) / view_height
                        bot_exc = (self._drlLastClip[1] - view_range.top()) / view_height
                        if top_exc >= limit / hyst and top_exc <= limit * hyst and (bot_exc >= limit / hyst) and (bot_exc <= limit * hyst):
                            x = self._datasetDisplay.x
                            y = self._datasetDisplay.y
                            cache_is_good = True
                    if not cache_is_good:
                        min_val = view_range.bottom() - limit * view_height
                        max_val = view_range.top() + limit * view_height
                        y = fn.clip_array(y, min_val, max_val)
                        self._drlLastClip = (min_val, max_val)
    self._datasetDisplay = PlotDataset(x, y, xAllFinite, yAllFinite)
    self.setProperty('xViewRangeWasChanged', False)
    self.setProperty('yViewRangeWasChanged', False)
    return self._datasetDisplay