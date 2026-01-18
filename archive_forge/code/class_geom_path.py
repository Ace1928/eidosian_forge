from __future__ import annotations
import typing
from collections import Counter
from contextlib import suppress
from warnings import warn
import numpy as np
from .._utils import SIZE_FACTOR, make_line_segments, match, to_rgba
from ..doctools import document
from ..exceptions import PlotnineWarning
from .geom import geom
@document
class geom_path(geom):
    """
    Connected points

    {usage}

    Parameters
    ----------
    {common_parameters}
    lineend : Literal["butt", "round", "projecting"], default="butt"
        Line end style. This option is applied for solid linetypes.
    linejoin : Literal["round", "miter", "bevel"], default="round"
        Line join style. This option is applied for solid linetypes.
    arrow : ~plotnine.geoms.geom_path.arrow, default=None
        Arrow specification. Default is no arrow.

    See Also
    --------
    plotnine.arrow : for adding arrowhead(s) to paths.
    """
    DEFAULT_AES = {'alpha': 1, 'color': 'black', 'linetype': 'solid', 'size': 0.5}
    REQUIRED_AES = {'x', 'y'}
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity', 'na_rm': False, 'lineend': 'butt', 'linejoin': 'round', 'arrow': None}

    def handle_na(self, data: pd.DataFrame) -> pd.DataFrame:

        def keep(x: Sequence[float]) -> npt.NDArray[np.bool_]:
            first = match([False], x, nomatch=1, start=0)[0]
            last = len(x) - match([False], x[::-1], nomatch=1, start=0)[0]
            bool_idx = np.hstack([np.repeat(False, first), np.repeat(True, last - first), np.repeat(False, len(x) - last)])
            return bool_idx
        bool_idx = data[['x', 'y', 'size', 'color', 'linetype']].isna().apply(keep, axis=0)
        bool_idx = np.all(bool_idx, axis=1)
        n1 = len(data)
        data = data[bool_idx]
        data.reset_index(drop=True, inplace=True)
        n2 = len(data)
        if n2 != n1 and (not self.params['na_rm']):
            msg = 'geom_path: Removed {} rows containing missing values.'
            warn(msg.format(n1 - n2), PlotnineWarning)
        return data

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: coord, ax: Axes, **params: Any):
        if not any(data['group'].duplicated()):
            warn('geom_path: Each group consist of only one observation. Do you need to adjust the group aesthetic?', PlotnineWarning)
        c = Counter(data['group'])
        counts = np.array([c[v] for v in data['group']])
        data = data[counts >= 2]
        if len(data) < 2:
            return
        data = data.sort_values('group', kind='mergesort')
        data.reset_index(drop=True, inplace=True)
        cols = {'color', 'size', 'linetype', 'alpha', 'group'}
        cols = cols & set(data.columns)
        num_unique_rows = len(data.drop_duplicates(cols))
        ngroup = len(np.unique(data['group'].to_numpy()))
        constant = num_unique_rows == ngroup
        params['constant'] = constant
        if not constant:
            self.draw_group(data, panel_params, coord, ax, **params)
        else:
            for _, gdata in data.groupby('group'):
                gdata.reset_index(inplace=True, drop=True)
                self.draw_group(gdata, panel_params, coord, ax, **params)

    @staticmethod
    def draw_group(data: pd.DataFrame, panel_params: panel_view, coord: coord, ax: Axes, **params: Any):
        data = coord.transform(data, panel_params, munch=True)
        data['size'] *= SIZE_FACTOR
        if 'constant' in params:
            constant: bool = params.pop('constant')
        else:
            constant = len(np.unique(data['group'].to_numpy())) == 1
        if not constant:
            _draw_segments(data, ax, **params)
        else:
            _draw_lines(data, ax, **params)
        if 'arrow' in params and params['arrow']:
            params['arrow'].draw(data, panel_params, coord, ax, constant=constant, **params)

    @staticmethod
    def draw_legend(data: pd.Series[Any], da: DrawingArea, lyr: layer) -> DrawingArea:
        """
        Draw a horizontal line in the box

        Parameters
        ----------
        data : Series
            Data Row
        da : DrawingArea
            Canvas
        lyr : layer
            Layer

        Returns
        -------
        out : DrawingArea
        """
        from matplotlib.lines import Line2D
        data['size'] *= SIZE_FACTOR
        x = [0, da.width]
        y = [0.5 * da.height] * 2
        color = to_rgba(data['color'], data['alpha'])
        key = Line2D(x, y, linestyle=data['linetype'], linewidth=data['size'], color=color, solid_capstyle='butt', antialiased=False)
        da.add_artist(key)
        return da

    @staticmethod
    def legend_key_size(data: pd.Series[Any], min_size: TupleInt2, lyr: layer) -> TupleInt2:
        w, h = min_size
        pad_w, pad_h = (w * 0.5, h * 0.5)
        _w = _h = data.get('size', 0) * SIZE_FACTOR
        if data['color'] is not None:
            w = max(w, _w + pad_w)
            h = max(h, _h + pad_h)
        return (w, h)