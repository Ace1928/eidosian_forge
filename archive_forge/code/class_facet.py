from __future__ import annotations
import itertools
import types
import typing
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from .._utils import cross_join, match
from ..exceptions import PlotnineError
from ..scales.scales import Scales
from .strips import Strips
class facet:
    """
    Base class for all facets

    Parameters
    ----------
    scales :
        Whether `x` or `y` scales should be allowed (free)
        to vary according to the data on each of the panel.
    shrink :
        Whether to shrink the scales to the output of the
        statistics instead of the raw data. Default is `True`.
    labeller :
        How to label the facets. A string value if it should be
        one of `["label_value", "label_both", "label_context"]`{.py}.
    as_table :
        If `True`, the facets are laid out like a table with
        the highest values at the bottom-right. If `False`
        the facets are laid out like a plot with the highest
        value a the top-right
    drop :
        If `True`, all factor levels not used in the data
        will automatically be dropped. If `False`, all
        factor levels will be shown, regardless of whether
        or not they appear in the data.
    dir :
        Direction in which to layout the panels. `h` for
        horizontal and `v` for vertical.
    """
    ncol: int
    nrow: int
    as_table = True
    drop = True
    shrink = True
    free: dict[Literal['x', 'y'], bool]
    params: dict[str, Any]
    theme: theme
    figure: Figure
    coordinates: coord
    layout: Layout
    axs: list[Axes]
    plot: ggplot
    strips: Strips
    grid_spec: GridSpec
    environment: Environment

    def __init__(self, scales: Literal['fixed', 'free', 'free_x', 'free_y']='fixed', shrink: bool=True, labeller: CanBeStripLabellingFunc='label_value', as_table: bool=True, drop: bool=True, dir: Literal['h', 'v']='h'):
        from .labelling import as_labeller
        self.shrink = shrink
        self.labeller = as_labeller(labeller)
        self.as_table = as_table
        self.drop = drop
        self.dir = dir
        self.free = {'x': scales in ('free_x', 'free'), 'y': scales in ('free_y', 'free')}

    def __radd__(self, plot: ggplot) -> ggplot:
        """
        Add facet to ggplot object
        """
        plot.facet = copy(self)
        plot.facet.environment = plot.environment
        return plot

    def setup(self, plot: ggplot):
        self.plot = plot
        self.layout = plot.layout
        if hasattr(plot, 'figure'):
            self.figure, self.axs = (plot.figure, plot.axs)
        else:
            self.figure, self.axs = self.make_figure()
        self.coordinates = plot.coordinates
        self.theme = plot.theme
        self.layout.axs = self.axs
        self.strips = Strips.from_facet(self)
        return (self.figure, self.axs)

    def setup_data(self, data: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Allow the facet to manipulate the data

        Parameters
        ----------
        data :
            Data for each of the layers

        Returns
        -------
        :
            Data for each of the layers

        Notes
        -----
        This method will be called after [](`~plotnine.facet.setup_params`),
        therefore the `params` property will be set.
        """
        return data

    def setup_params(self, data: list[pd.DataFrame]):
        """
        Create facet parameters

        Parameters
        ----------
        data :
            Plot data and data for the layers
        """
        self.params = {}

    def init_scales(self, layout: pd.DataFrame, x_scale: Optional[scale]=None, y_scale: Optional[scale]=None) -> types.SimpleNamespace:
        scales = types.SimpleNamespace()
        if x_scale is not None:
            n = layout['SCALE_X'].max()
            scales.x = Scales([x_scale.clone() for i in range(n)])
        if y_scale is not None:
            n = layout['SCALE_Y'].max()
            scales.y = Scales([y_scale.clone() for i in range(n)])
        return scales

    def map(self, data: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
        """
        Assign a data points to panels

        Parameters
        ----------
        data :
            Data for a layer
        layout :
            As returned by self.compute_layout

        Returns
        -------
        :
            Data with all points mapped to the panels
            on which they will be plotted.
        """
        msg = '{} should implement this method.'
        raise NotImplementedError(msg.format(self.__class__.__name__))

    def compute_layout(self, data: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Compute layout

        Parameters
        ----------
        data :
            Dataframe for a each layer
        """
        msg = '{} should implement this method.'
        raise NotImplementedError(msg.format(self.__class__.__name__))

    def finish_data(self, data: pd.DataFrame, layout: Layout) -> pd.DataFrame:
        """
        Modify data before it is drawn out by the geom

        The default is to return the data without modification.
        Subclasses should override this method as the require.

        Parameters
        ----------
        data :
            A single layer's data.
        layout :
            Layout

        Returns
        -------
        :
            Modified layer data
        """
        return data

    def train_position_scales(self, layout: Layout, layers: Layers) -> facet:
        """
        Compute ranges for the x and y scales
        """
        _layout = layout.layout
        panel_scales_x = layout.panel_scales_x
        panel_scales_y = layout.panel_scales_y
        for layer in layers:
            data = layer.data
            match_id = match(data['PANEL'], _layout['PANEL'])
            if panel_scales_x:
                x_vars = list(set(panel_scales_x[0].aesthetics) & set(data.columns))
                SCALE_X = _layout['SCALE_X'].iloc[match_id].tolist()
                panel_scales_x.train(data, x_vars, SCALE_X)
            if panel_scales_y:
                y_vars = list(set(panel_scales_y[0].aesthetics) & set(data.columns))
                SCALE_Y = _layout['SCALE_Y'].iloc[match_id].tolist()
                panel_scales_y.train(data, y_vars, SCALE_Y)
        return self

    def make_strips(self, layout_info: layout_details, ax: Axes) -> Strips:
        """
        Create strips for the facet

        Parameters
        ----------
        layout_info :
            Layout information. Row from the layout table

        ax :
            Axes to label
        """
        return Strips()

    def set_limits_breaks_and_labels(self, panel_params: panel_view, ax: Axes):
        """
        Add limits, breaks and labels to the axes

        Parameters
        ----------
        panel_params :
            range information for the axes
        ax :
            Axes
        """
        from .._mpl.ticker import MyFixedFormatter

        def _inf_to_none(t: tuple[float, float]) -> tuple[float | None, float | None]:
            """
            Replace infinities with None
            """
            a = t[0] if np.isfinite(t[0]) else None
            b = t[1] if np.isfinite(t[1]) else None
            return (a, b)
        theme = self.theme
        ax.set_xlim(*_inf_to_none(panel_params.x.range))
        ax.set_ylim(*_inf_to_none(panel_params.y.range))
        if typing.TYPE_CHECKING:
            assert callable(ax.set_xticks)
            assert callable(ax.set_yticks)
        ax.set_xticks(panel_params.x.breaks, panel_params.x.labels)
        ax.set_yticks(panel_params.y.breaks, panel_params.y.labels)
        ax.set_xticks(panel_params.x.minor_breaks, minor=True)
        ax.set_yticks(panel_params.y.minor_breaks, minor=True)
        ax.xaxis.set_major_formatter(MyFixedFormatter(panel_params.x.labels))
        ax.yaxis.set_major_formatter(MyFixedFormatter(panel_params.y.labels))
        margin = theme.getp(('axis_text_x', 'margin'))
        pad_x = margin.get_as('t', 'pt')
        margin = theme.getp(('axis_text_y', 'margin'))
        pad_y = margin.get_as('r', 'pt')
        ax.tick_params(axis='x', which='major', pad=pad_x)
        ax.tick_params(axis='y', which='major', pad=pad_y)

    def __deepcopy__(self, memo: dict[Any, Any]) -> facet:
        """
        Deep copy without copying the dataframe and environment
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        old = self.__dict__
        new = result.__dict__
        shallow = {'figure', 'axs', 'first_ax', 'last_ax'}
        for key, item in old.items():
            if key in shallow:
                new[key] = item
                memo[id(new[key])] = new[key]
            else:
                new[key] = deepcopy(item, memo)
        return result

    def _make_figure(self) -> tuple[Figure, GridSpec]:
        """
        Create figure & gridspec
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        return (plt.figure(), GridSpec(self.nrow, self.ncol))

    def make_figure(self) -> tuple[Figure, list[Axes]]:
        """
        Create and return Matplotlib figure and subplot axes
        """
        num_panels = len(self.layout.layout)
        axsarr = np.empty((self.nrow, self.ncol), dtype=object)
        figure, gs = self._make_figure()
        self.grid_spec = gs
        it = itertools.product(range(self.nrow), range(self.ncol))
        for i, (row, col) in enumerate(it):
            axsarr[row, col] = figure.add_subplot(gs[i])
        if self.dir == 'h':
            order: Literal['C', 'F'] = 'C'
            if not self.as_table:
                axsarr = axsarr[::-1]
        elif self.dir == 'v':
            order = 'F'
            if not self.as_table:
                axsarr = np.array([row[::-1] for row in axsarr])
        else:
            raise ValueError(f'Bad value `dir="{self.dir}"` for direction')
        axs = axsarr.ravel(order)
        for ax in axs[num_panels:]:
            figure.delaxes(ax)
        axs = axs[:num_panels]
        return (figure, list(axs))

    def _aspect_ratio(self) -> Optional[float]:
        """
        Return the aspect_ratio
        """
        aspect_ratio = self.theme.getp('aspect_ratio')
        if aspect_ratio == 'auto':
            if not self.free['x'] and (not self.free['y']):
                aspect_ratio = self.coordinates.aspect(self.layout.panel_params[0])
            else:
                aspect_ratio = None
        return aspect_ratio