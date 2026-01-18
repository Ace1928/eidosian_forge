from __future__ import annotations
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
def _make_plot(self, fig: Figure) -> None:
    colors = self._get_colors()
    stacking_id = self._get_stacking_id()
    data = create_iter_data_given_by(self.data, self._kind) if self.by is not None else self.data
    for i, (label, y) in enumerate(self._iter_data(data=data)):
        ax = self._get_ax(i)
        kwds = self.kwds.copy()
        if self.color is not None:
            kwds['color'] = self.color
        label = pprint_thing(label)
        label = self._mark_right_label(label, index=i)
        kwds['label'] = label
        style, kwds = self._apply_style_colors(colors, kwds, i, label)
        if style is not None:
            kwds['style'] = style
        self._make_plot_keywords(kwds, y)
        if self.by is not None:
            kwds['bins'] = kwds['bins'][i]
            kwds['label'] = self.columns
            kwds.pop('color')
        if self.weights is not None:
            kwds['weights'] = type(self)._get_column_weights(self.weights, i, y)
        y = reformat_hist_y_given_by(y, self.by)
        artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)
        if self.by is not None:
            ax.set_title(pprint_thing(label))
        self._append_legend_handles_labels(artists[0], label)