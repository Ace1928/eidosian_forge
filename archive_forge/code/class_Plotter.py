from __future__ import annotations
import io
import os
import re
import inspect
import itertools
import textwrap
from contextlib import contextmanager
from collections import abc
from collections.abc import Callable, Generator
from typing import Any, List, Literal, Optional, cast
from xml.etree import ElementTree
from cycler import cycler
import pandas as pd
from pandas import DataFrame, Series, Index
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move
from seaborn._core.scales import Scale
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.properties import PROPERTIES, Property
from seaborn._core.typing import (
from seaborn._core.exceptions import PlotSpecError
from seaborn._core.rules import categorical_order
from seaborn._compat import get_layout_engine, set_layout_engine
from seaborn.utils import _version_predates
from seaborn.rcmod import axes_style, plotting_context
from seaborn.palettes import color_palette
from typing import TYPE_CHECKING, TypedDict
class Plotter:
    """
    Engine for compiling a :class:`Plot` spec into a Matplotlib figure.

    This class is not intended to be instantiated directly by users.

    """
    _data: PlotData
    _layers: list[Layer]
    _figure: Figure

    def __init__(self, pyplot: bool, theme: dict[str, Any]):
        self._pyplot = pyplot
        self._theme = theme
        self._legend_contents: list[tuple[tuple[str, str | int], list[Artist], list[str]]] = []
        self._scales: dict[str, Scale] = {}

    def save(self, loc, **kwargs) -> Plotter:
        kwargs.setdefault('dpi', 96)
        try:
            loc = os.path.expanduser(loc)
        except TypeError:
            pass
        self._figure.savefig(loc, **kwargs)
        return self

    def show(self, **kwargs) -> None:
        """
        Display the plot by hooking into pyplot.

        This method calls :func:`matplotlib.pyplot.show` with any keyword parameters.

        """
        import matplotlib.pyplot as plt
        with theme_context(self._theme):
            plt.show(**kwargs)

    def _repr_png_(self) -> tuple[bytes, dict[str, float]] | None:
        if Plot.config.display['format'] != 'png':
            return None
        buffer = io.BytesIO()
        factor = 2 if Plot.config.display['hidpi'] else 1
        scaling = Plot.config.display['scaling'] / factor
        dpi = 96 * factor
        with theme_context(self._theme):
            self._figure.savefig(buffer, dpi=dpi, format='png', bbox_inches='tight')
        data = buffer.getvalue()
        w, h = Image.open(buffer).size
        metadata = {'width': w * scaling, 'height': h * scaling}
        return (data, metadata)

    def _repr_svg_(self) -> str | None:
        if Plot.config.display['format'] != 'svg':
            return None
        scaling = Plot.config.display['scaling']
        buffer = io.StringIO()
        with theme_context(self._theme):
            self._figure.savefig(buffer, format='svg', bbox_inches='tight')
        root = ElementTree.fromstring(buffer.getvalue())
        w = scaling * float(root.attrib['width'][:-2])
        h = scaling * float(root.attrib['height'][:-2])
        root.attrib.update(width=f'{w}pt', height=f'{h}pt', viewbox=f'0 0 {w} {h}')
        ElementTree.ElementTree(root).write((out := io.BytesIO()))
        return out.getvalue().decode()

    def _extract_data(self, p: Plot) -> tuple[PlotData, list[Layer]]:
        common_data = p._data.join(None, p._facet_spec.get('variables')).join(None, p._pair_spec.get('variables'))
        layers: list[Layer] = []
        for layer in p._layers:
            spec = layer.copy()
            spec['data'] = common_data.join(layer.get('source'), layer.get('vars'))
            layers.append(spec)
        return (common_data, layers)

    def _resolve_label(self, p: Plot, var: str, auto_label: str | None) -> str:
        if re.match('[xy]\\d+', var):
            key = var if var in p._labels else var[0]
        else:
            key = var
        label: str
        if key in p._labels:
            manual_label = p._labels[key]
            if callable(manual_label) and auto_label is not None:
                label = manual_label(auto_label)
            else:
                label = cast(str, manual_label)
        elif auto_label is None:
            label = ''
        else:
            label = auto_label
        return label

    def _setup_figure(self, p: Plot, common: PlotData, layers: list[Layer]) -> None:
        subplot_spec = p._subplot_spec.copy()
        facet_spec = p._facet_spec.copy()
        pair_spec = p._pair_spec.copy()
        for axis in 'xy':
            if axis in p._shares:
                subplot_spec[f'share{axis}'] = p._shares[axis]
        for dim in ['col', 'row']:
            if dim in common.frame and dim not in facet_spec['structure']:
                order = categorical_order(common.frame[dim])
                facet_spec['structure'][dim] = order
        self._subplots = subplots = Subplots(subplot_spec, facet_spec, pair_spec)
        self._figure = subplots.init_figure(pair_spec, self._pyplot, p._figure_spec, p._target)
        for sub in subplots:
            ax = sub['ax']
            for axis in 'xy':
                axis_key = sub[axis]
                names = [common.names.get(axis_key), *(layer['data'].names.get(axis_key) for layer in layers)]
                auto_label = next((name for name in names if name is not None), None)
                label = self._resolve_label(p, axis_key, auto_label)
                ax.set(**{f'{axis}label': label})
                axis_obj = getattr(ax, f'{axis}axis')
                visible_side = {'x': 'bottom', 'y': 'left'}.get(axis)
                show_axis_label = sub[visible_side] or not p._pair_spec.get('cross', True) or (axis in p._pair_spec.get('structure', {}) and bool(p._pair_spec.get('wrap')))
                axis_obj.get_label().set_visible(show_axis_label)
                show_tick_labels = show_axis_label or subplot_spec.get(f'share{axis}') not in (True, 'all', {'x': 'col', 'y': 'row'}[axis])
                for group in ('major', 'minor'):
                    side = {'x': 'bottom', 'y': 'left'}[axis]
                    axis_obj.set_tick_params(**{f'label{side}': show_tick_labels})
                    for t in getattr(axis_obj, f'get_{group}ticklabels')():
                        t.set_visible(show_tick_labels)
            title_parts = []
            for dim in ['col', 'row']:
                if sub[dim] is not None:
                    val = self._resolve_label(p, 'title', f'{sub[dim]}')
                    if dim in p._labels:
                        key = self._resolve_label(p, dim, common.names.get(dim))
                        val = f'{key} {val}'
                    title_parts.append(val)
            has_col = sub['col'] is not None
            has_row = sub['row'] is not None
            show_title = has_col and has_row or ((has_col or has_row) and p._facet_spec.get('wrap')) or (has_col and sub['top']) or has_row
            if title_parts:
                title = ' | '.join(title_parts)
                title_text = ax.set_title(title)
                title_text.set_visible(show_title)
            elif not (has_col or has_row):
                title = self._resolve_label(p, 'title', None)
                title_text = ax.set_title(title)

    def _compute_stats(self, spec: Plot, layers: list[Layer]) -> None:
        grouping_vars = [v for v in PROPERTIES if v not in 'xy']
        grouping_vars += ['col', 'row', 'group']
        pair_vars = spec._pair_spec.get('structure', {})
        for layer in layers:
            data = layer['data']
            mark = layer['mark']
            stat = layer['stat']
            if stat is None:
                continue
            iter_axes = itertools.product(*[pair_vars.get(axis, [axis]) for axis in 'xy'])
            old = data.frame
            if pair_vars:
                data.frames = {}
                data.frame = data.frame.iloc[:0]
            for coord_vars in iter_axes:
                pairings = ('xy', coord_vars)
                df = old.copy()
                scales = self._scales.copy()
                for axis, var in zip(*pairings):
                    if axis != var:
                        df = df.rename(columns={var: axis})
                        drop_cols = [x for x in df if re.match(f'{axis}\\d+', str(x))]
                        df = df.drop(drop_cols, axis=1)
                        scales[axis] = scales[var]
                orient = layer['orient'] or mark._infer_orient(scales)
                if stat.group_by_orient:
                    grouper = [orient, *grouping_vars]
                else:
                    grouper = grouping_vars
                groupby = GroupBy(grouper)
                res = stat(df, groupby, orient, scales)
                if pair_vars:
                    data.frames[coord_vars] = res
                else:
                    data.frame = res

    def _get_scale(self, p: Plot, var: str, prop: Property, values: Series) -> Scale:
        if re.match('[xy]\\d+', var):
            key = var if var in p._scales else var[0]
        else:
            key = var
        if key in p._scales:
            arg = p._scales[key]
            if arg is None or isinstance(arg, Scale):
                scale = arg
            else:
                scale = prop.infer_scale(arg, values)
        else:
            scale = prop.default_scale(values)
        return scale

    def _get_subplot_data(self, df, var, view, share_state):
        if share_state in [True, 'all']:
            seed_values = df[var]
        else:
            if share_state in [False, 'none']:
                idx = self._get_subplot_index(df, view)
            elif share_state in df:
                use_rows = df[share_state] == view[share_state]
                idx = df.index[use_rows]
            else:
                idx = df.index
            seed_values = df.loc[idx, var]
        return seed_values

    def _setup_scales(self, p: Plot, common: PlotData, layers: list[Layer], variables: list[str] | None=None) -> None:
        if variables is None:
            variables = []
            for layer in layers:
                variables.extend(layer['data'].frame.columns)
                for df in layer['data'].frames.values():
                    variables.extend((str(v) for v in df if v not in variables))
            variables = [v for v in variables if v not in self._scales]
        for var in variables:
            m = re.match('^(?P<coord>(?P<axis>x|y)\\d*).*', var)
            if m is None:
                coord = axis = None
            else:
                coord = m['coord']
                axis = m['axis']
            prop_key = var if axis is None else axis
            scale_key = var if coord is None else coord
            if prop_key not in PROPERTIES:
                continue
            cols = [var, 'col', 'row']
            parts = [common.frame.filter(cols)]
            for layer in layers:
                parts.append(layer['data'].frame.filter(cols))
                for df in layer['data'].frames.values():
                    parts.append(df.filter(cols))
            var_df = pd.concat(parts, ignore_index=True)
            prop = PROPERTIES[prop_key]
            scale = self._get_scale(p, scale_key, prop, var_df[var])
            if scale_key not in p._variables:
                scale._priority = 0
            if axis is None:
                share_state = None
                subplots = []
            else:
                share_state = self._subplots.subplot_spec[f'share{axis}']
                subplots = [view for view in self._subplots if view[axis] == coord]
            if scale is None:
                self._scales[var] = Scale._identity()
            else:
                try:
                    self._scales[var] = scale._setup(var_df[var], prop)
                except Exception as err:
                    raise PlotSpecError._during('Scale setup', var) from err
            if axis is None or (var != coord and coord in p._variables):
                continue
            transformed_data = []
            for layer in layers:
                index = layer['data'].frame.index
                empty_series = pd.Series(dtype=float, index=index, name=var)
                transformed_data.append(empty_series)
            for view in subplots:
                axis_obj = getattr(view['ax'], f'{axis}axis')
                seed_values = self._get_subplot_data(var_df, var, view, share_state)
                view_scale = scale._setup(seed_values, prop, axis=axis_obj)
                view['ax'].set(**{f'{axis}scale': view_scale._matplotlib_scale})
                for layer, new_series in zip(layers, transformed_data):
                    layer_df = layer['data'].frame
                    if var not in layer_df:
                        continue
                    idx = self._get_subplot_index(layer_df, view)
                    try:
                        new_series.loc[idx] = view_scale(layer_df.loc[idx, var])
                    except Exception as err:
                        spec_error = PlotSpecError._during('Scaling operation', var)
                        raise spec_error from err
            for layer, new_series in zip(layers, transformed_data):
                layer_df = layer['data'].frame
                if var in layer_df:
                    layer_df[var] = pd.to_numeric(new_series)

    def _plot_layer(self, p: Plot, layer: Layer) -> None:
        data = layer['data']
        mark = layer['mark']
        move = layer['move']
        default_grouping_vars = ['col', 'row', 'group']
        grouping_properties = [v for v in PROPERTIES if v[0] not in 'xy']
        pair_variables = p._pair_spec.get('structure', {})
        for subplots, df, scales in self._generate_pairings(data, pair_variables):
            orient = layer['orient'] or mark._infer_orient(scales)

            def get_order(var):
                if var not in 'xy' and var in scales:
                    return getattr(scales[var], 'order', None)
            if orient in df:
                width = pd.Series(index=df.index, dtype=float)
                for view in subplots:
                    view_idx = self._get_subplot_data(df, orient, view, p._shares.get(orient)).index
                    view_df = df.loc[view_idx]
                    if 'width' in mark._mappable_props:
                        view_width = mark._resolve(view_df, 'width', None)
                    elif 'width' in df:
                        view_width = view_df['width']
                    else:
                        view_width = 0.8
                    spacing = scales[orient]._spacing(view_df.loc[view_idx, orient])
                    width.loc[view_idx] = view_width * spacing
                df['width'] = width
            if 'baseline' in mark._mappable_props:
                baseline = mark._resolve(df, 'baseline', None)
            else:
                baseline = 0 if 'baseline' not in df else df['baseline']
            df['baseline'] = baseline
            if move is not None:
                moves = move if isinstance(move, list) else [move]
                for move_step in moves:
                    move_by = getattr(move_step, 'by', None)
                    if move_by is None:
                        move_by = grouping_properties
                    move_groupers = [*move_by, *default_grouping_vars]
                    if move_step.group_by_orient:
                        move_groupers.insert(0, orient)
                    order = {var: get_order(var) for var in move_groupers}
                    groupby = GroupBy(order)
                    df = move_step(df, groupby, orient, scales)
            df = self._unscale_coords(subplots, df, orient)
            grouping_vars = mark._grouping_props + default_grouping_vars
            split_generator = self._setup_split_generator(grouping_vars, df, subplots)
            mark._plot(split_generator, scales, orient)
        for view in self._subplots:
            view['ax'].autoscale_view()
        if layer['legend']:
            self._update_legend_contents(p, mark, data, scales, layer['label'])

    def _unscale_coords(self, subplots: list[dict], df: DataFrame, orient: str) -> DataFrame:
        coord_cols = [c for c in df if re.match('^[xy]\\D*$', str(c))]
        out_df = df.drop(coord_cols, axis=1).reindex(df.columns, axis=1).copy(deep=False)
        for view in subplots:
            view_df = self._filter_subplot_data(df, view)
            axes_df = view_df[coord_cols]
            for var, values in axes_df.items():
                axis = getattr(view['ax'], f'{str(var)[0]}axis')
                transform = axis.get_transform().inverted().transform
                inverted = transform(values)
                out_df.loc[values.index, str(var)] = inverted
        return out_df

    def _generate_pairings(self, data: PlotData, pair_variables: dict) -> Generator[tuple[list[dict], DataFrame, dict[str, Scale]], None, None]:
        iter_axes = itertools.product(*[pair_variables.get(axis, [axis]) for axis in 'xy'])
        for x, y in iter_axes:
            subplots = []
            for view in self._subplots:
                if view['x'] == x and view['y'] == y:
                    subplots.append(view)
            if data.frame.empty and data.frames:
                out_df = data.frames[x, y].copy()
            elif not pair_variables:
                out_df = data.frame.copy()
            elif data.frame.empty and data.frames:
                out_df = data.frames[x, y].copy()
            else:
                out_df = data.frame.copy()
            scales = self._scales.copy()
            if x in out_df:
                scales['x'] = self._scales[x]
            if y in out_df:
                scales['y'] = self._scales[y]
            for axis, var in zip('xy', (x, y)):
                if axis != var:
                    out_df = out_df.rename(columns={var: axis})
                    cols = [col for col in out_df if re.match(f'{axis}\\d+', str(col))]
                    out_df = out_df.drop(cols, axis=1)
            yield (subplots, out_df, scales)

    def _get_subplot_index(self, df: DataFrame, subplot: dict) -> Index:
        dims = df.columns.intersection(['col', 'row'])
        if dims.empty:
            return df.index
        keep_rows = pd.Series(True, df.index, dtype=bool)
        for dim in dims:
            keep_rows &= df[dim] == subplot[dim]
        return df.index[keep_rows]

    def _filter_subplot_data(self, df: DataFrame, subplot: dict) -> DataFrame:
        dims = df.columns.intersection(['col', 'row'])
        if dims.empty:
            return df
        keep_rows = pd.Series(True, df.index, dtype=bool)
        for dim in dims:
            keep_rows &= df[dim] == subplot[dim]
        return df[keep_rows]

    def _setup_split_generator(self, grouping_vars: list[str], df: DataFrame, subplots: list[dict[str, Any]]) -> Callable[[], Generator]:
        grouping_keys = []
        grouping_vars = [v for v in grouping_vars if v in df and v not in ['col', 'row']]
        for var in grouping_vars:
            order = getattr(self._scales[var], 'order', None)
            if order is None:
                order = categorical_order(df[var])
            grouping_keys.append(order)

        def split_generator(keep_na=False) -> Generator:
            for view in subplots:
                axes_df = self._filter_subplot_data(df, view)
                axes_df_inf_as_nan = axes_df.copy()
                axes_df_inf_as_nan = axes_df_inf_as_nan.mask(axes_df_inf_as_nan.isin([np.inf, -np.inf]), np.nan)
                if keep_na:
                    present = axes_df_inf_as_nan.notna().all(axis=1)
                    nulled = {}
                    for axis in 'xy':
                        if axis in axes_df:
                            nulled[axis] = axes_df[axis].where(present)
                    axes_df = axes_df_inf_as_nan.assign(**nulled)
                else:
                    axes_df = axes_df_inf_as_nan.dropna()
                subplot_keys = {}
                for dim in ['col', 'row']:
                    if view[dim] is not None:
                        subplot_keys[dim] = view[dim]
                if not grouping_vars or not any(grouping_keys):
                    if not axes_df.empty:
                        yield (subplot_keys, axes_df.copy(), view['ax'])
                    continue
                grouped_df = axes_df.groupby(grouping_vars, sort=False, as_index=False, observed=False)
                for key in itertools.product(*grouping_keys):
                    pd_key = key[0] if len(key) == 1 and _version_predates(pd, '2.2.0') else key
                    try:
                        df_subset = grouped_df.get_group(pd_key)
                    except KeyError:
                        df_subset = axes_df.loc[[]]
                    if df_subset.empty:
                        continue
                    sub_vars = dict(zip(grouping_vars, key))
                    sub_vars.update(subplot_keys)
                    yield (sub_vars, df_subset.copy(), view['ax'])
        return split_generator

    def _update_legend_contents(self, p: Plot, mark: Mark, data: PlotData, scales: dict[str, Scale], layer_label: str | None) -> None:
        """Add legend artists / labels for one layer in the plot."""
        if data.frame.empty and data.frames:
            legend_vars: list[str] = []
            for frame in data.frames.values():
                frame_vars = frame.columns.intersection(list(scales))
                legend_vars.extend((v for v in frame_vars if v not in legend_vars))
        else:
            legend_vars = list(data.frame.columns.intersection(list(scales)))
        if layer_label is not None:
            legend_title = str(p._labels.get('legend', ''))
            layer_key = (legend_title, -1)
            artist = mark._legend_artist([], None, {})
            if artist is not None:
                for content in self._legend_contents:
                    if content[0] == layer_key:
                        content[1].append(artist)
                        content[2].append(layer_label)
                        break
                else:
                    self._legend_contents.append((layer_key, [artist], [layer_label]))
        schema: list[tuple[tuple[str, str | int], list[str], tuple[list[Any], list[str]]]] = []
        schema = []
        for var in legend_vars:
            var_legend = scales[var]._legend
            if var_legend is not None:
                values, labels = var_legend
                for (_, part_id), part_vars, _ in schema:
                    if data.ids[var] == part_id:
                        part_vars.append(var)
                        break
                else:
                    title = self._resolve_label(p, var, data.names[var])
                    entry = ((title, data.ids[var]), [var], (values, labels))
                    schema.append(entry)
        contents: list[tuple[tuple[str, str | int], Any, list[str]]] = []
        for key, variables, (values, labels) in schema:
            artists = []
            for val in values:
                artist = mark._legend_artist(variables, val, scales)
                if artist is not None:
                    artists.append(artist)
            if artists:
                contents.append((key, artists, labels))
        self._legend_contents.extend(contents)

    def _make_legend(self, p: Plot) -> None:
        """Create the legend artist(s) and add onto the figure."""
        merged_contents: dict[tuple[str, str | int], tuple[list[tuple[Artist, ...]], list[str]]] = {}
        for key, new_artists, labels in self._legend_contents:
            if key not in merged_contents:
                new_artist_tuples = [tuple([a]) for a in new_artists]
                merged_contents[key] = (new_artist_tuples, labels)
            else:
                existing_artists = merged_contents[key][0]
                for i, new_artist in enumerate(new_artists):
                    existing_artists[i] += tuple([new_artist])
        loc = 'center right' if self._pyplot else 'center left'
        base_legend = None
        for (name, _), (handles, labels) in merged_contents.items():
            legend = mpl.legend.Legend(self._figure, handles, labels, title=name, loc=loc, bbox_to_anchor=(0.98, 0.55))
            if base_legend:
                base_legend_box = base_legend.get_children()[0]
                this_legend_box = legend.get_children()[0]
                base_legend_box.get_children().extend(this_legend_box.get_children())
            else:
                base_legend = legend
                self._figure.legends.append(legend)

    def _finalize_figure(self, p: Plot) -> None:
        for sub in self._subplots:
            ax = sub['ax']
            for axis in 'xy':
                axis_key = sub[axis]
                axis_obj = getattr(ax, f'{axis}axis')
                if axis_key in p._limits or axis in p._limits:
                    convert_units = getattr(ax, f'{axis}axis').convert_units
                    a, b = p._limits.get(axis_key) or p._limits[axis]
                    lo = a if a is None else convert_units(a)
                    hi = b if b is None else convert_units(b)
                    if isinstance(a, str):
                        lo = cast(float, lo) - 0.5
                    if isinstance(b, str):
                        hi = cast(float, hi) + 0.5
                    ax.set(**{f'{axis}lim': (lo, hi)})
                if axis_key in self._scales:
                    self._scales[axis_key]._finalize(p, axis_obj)
        if (engine_name := p._layout_spec.get('engine', default)) is not default:
            set_layout_engine(self._figure, engine_name)
        elif p._target is None:
            set_layout_engine(self._figure, 'tight')
        if (extent := p._layout_spec.get('extent')) is not None:
            engine = get_layout_engine(self._figure)
            if engine is None:
                self._figure.subplots_adjust(*extent)
            else:
                left, bottom, right, top = extent
                width, height = (right - left, top - bottom)
                try:
                    engine.set(rect=[left, bottom, width, height])
                except TypeError:
                    pass