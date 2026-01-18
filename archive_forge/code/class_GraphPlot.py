from collections import defaultdict
import numpy as np
import param
from bokeh.models import (
from ...core.data import Dataset
from ...core.options import Cycle, abbreviated_exception
from ...core.util import dimension_sanitizer, unique_array
from ...util.transform import dim
from ..mixins import ChordMixin, GraphMixin
from ..util import get_directed_graph_paths, process_cmap
from .chart import ColorbarPlot, PointPlot
from .element import CompositeElementPlot, LegendPlot
from .styles import (
class GraphPlot(GraphMixin, CompositeElementPlot, ColorbarPlot, LegendPlot):
    arrowhead_length = param.Number(default=0.025, doc='\n      If directed option is enabled this determines the length of the\n      arrows as fraction of the overall extent of the graph.')
    directed = param.Boolean(default=False, doc='\n      Whether to draw arrows on the graph edges to indicate the\n      directionality of each edge.')
    selection_policy = param.ObjectSelector(default='nodes', objects=['edges', 'nodes', None], doc='\n        Determines policy for inspection of graph components, i.e. whether to highlight\n        nodes or edges when selecting connected edges and nodes respectively.')
    inspection_policy = param.ObjectSelector(default='nodes', objects=['edges', 'nodes', None], doc='\n        Determines policy for inspection of graph components, i.e. whether to highlight\n        nodes or edges when hovering over connected edges and nodes respectively.')
    tools = param.List(default=['hover', 'tap'], doc='\n        A list of plugin tools to use on the plot.')
    color_index = param.ClassSelector(default=None, class_=(str, int), allow_None=True, doc="\n        Deprecated in favor of color style mapping, e.g. `node_color=dim('color')`")
    edge_color_index = param.ClassSelector(default=None, class_=(str, int), allow_None=True, doc="\n        Deprecated in favor of color style mapping, e.g. `edge_color=dim('color')`")
    _style_groups = {'scatter': 'node', 'multi_line': 'edge', 'patches': 'edge', 'bezier': 'edge'}
    style_opts = ['edge_' + p for p in base_properties + fill_properties + line_properties] + ['node_' + p for p in base_properties + fill_properties + line_properties] + ['node_size', 'cmap', 'edge_cmap', 'node_cmap', 'node_radius', 'node_marker']
    _nonvectorized_styles = base_properties + ['cmap', 'edge_cmap', 'node_cmap']
    filled = False
    bezier = False
    _node_columns = [0, 1]

    @property
    def edge_glyph(self):
        if self.filled:
            return 'patches_1'
        elif self.bezier:
            return 'bezier_1'
        else:
            return 'multi_line_1'

    def _hover_opts(self, element):
        if self.inspection_policy == 'nodes':
            dims = element.nodes.dimensions()
            dims = [(dims[2].pprint_label, '@{index_hover}')] + dims[3:]
        elif self.inspection_policy == 'edges':
            kdims = [(kd.pprint_label, '@{%s_values}' % kd) if kd in ('start', 'end') else kd for kd in element.kdims]
            dims = kdims + element.vdims
        else:
            dims = []
        return (dims, {})

    def _get_edge_colors(self, element, ranges, edge_data, edge_mapping, style):
        cdim = element.get_dimension(self.edge_color_index)
        if not cdim:
            return
        elstyle = self.lookup_options(element, 'style')
        cycle = elstyle.kwargs.get('edge_color')
        if not isinstance(cycle, Cycle):
            cycle = None
        idx = element.get_dimension_index(cdim)
        field = dimension_sanitizer(cdim.name)
        cvals = element.dimension_values(cdim)
        if idx in self._node_columns:
            factors = element.nodes.dimension_values(2, expanded=False)
        elif idx == 2 and cvals.dtype.kind in 'uif':
            factors = None
        else:
            factors = unique_array(cvals)
        default_cmap = 'viridis' if factors is None else 'tab20'
        cmap = style.get('edge_cmap', style.get('cmap', default_cmap))
        nan_colors = {k: rgba_tuple(v) for k, v in self.clipping_colors.items()}
        if factors is None or (factors.dtype.kind in 'uif' and idx not in self._node_columns):
            colors, factors = (None, None)
        else:
            if factors.dtype.kind == 'f':
                cvals = cvals.astype(np.int32)
                factors = factors.astype(np.int32)
            if factors.dtype.kind not in 'SU':
                field += '_str__'
                cvals = [str(f) for f in cvals]
                factors = (str(f) for f in factors)
            factors = list(factors)
            if isinstance(cmap, dict):
                colors = [cmap.get(f, nan_colors.get('NaN', self._default_nan)) for f in factors]
            else:
                colors = process_cmap(cycle or cmap, len(factors))
        if field not in edge_data:
            edge_data[field] = cvals
        edge_style = dict(style, cmap=cmap)
        mapper = self._get_colormapper(cdim, element, ranges, edge_style, factors, colors, 'edge', 'edge_colormapper')
        transform = {'field': field, 'transform': mapper}
        color_type = 'fill_color' if self.filled else 'line_color'
        edge_mapping['edge_' + color_type] = transform
        edge_mapping['edge_nonselection_' + color_type] = transform
        edge_mapping['edge_selection_' + color_type] = transform

    def _get_edge_paths(self, element, ranges):
        path_data, mapping = ({}, {})
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        if element._edgepaths is not None:
            edges = element._split_edgepaths.split(datatype='array', dimensions=element.edgepaths.kdims)
            if len(edges) == len(element):
                path_data['xs'] = [path[:, xidx] for path in edges]
                path_data['ys'] = [path[:, yidx] for path in edges]
                mapping = {'xs': 'xs', 'ys': 'ys'}
            else:
                raise ValueError('Edge paths do not match the number of supplied edges.Expected %d, found %d paths.' % (len(element), len(edges)))
        elif self.directed:
            xdim, ydim = element.nodes.kdims[:2]
            x_range = ranges[xdim.name]['combined']
            y_range = ranges[ydim.name]['combined']
            arrow_len = np.hypot(y_range[1] - y_range[0], x_range[1] - x_range[0]) * self.arrowhead_length
            arrows = get_directed_graph_paths(element, arrow_len)
            path_data['xs'] = [arr[:, 0] for arr in arrows]
            path_data['ys'] = [arr[:, 1] for arr in arrows]
        return (path_data, mapping)

    def get_data(self, element, ranges, style):
        static = self.static_source
        self.handles['static_source'] = static
        self.static_source = False
        nodes = element.nodes.dimension_values(2)
        node_positions = element.nodes.array([0, 1])
        if nodes.dtype.kind not in 'uif':
            node_indices = {v: i for i, v in enumerate(nodes)}
            index = np.array([node_indices[n] for n in nodes], dtype=np.int32)
            layout = {node_indices[k]: (y, x) if self.invert_axes else (x, y) for k, (x, y) in zip(nodes, node_positions)}
        else:
            index = nodes.astype(np.int32)
            layout = {k: (y, x) if self.invert_axes else (x, y) for k, (x, y) in zip(index, node_positions)}
        point_data = {'index': index}
        fixed_color = style.pop('node_color', None)
        cycle = self.lookup_options(element, 'style').kwargs.get('node_color')
        if isinstance(cycle, Cycle) and 'cmap' not in style:
            colors = cycle
        else:
            colors = None
        cdata, cmapping = self._get_color_data(element.nodes, ranges, style, name='node_fill_color', colors=colors, int_categories=True)
        if fixed_color is not None and (not cdata):
            style['node_color'] = fixed_color
        point_data.update(cdata)
        point_mapping = cmapping
        if 'node_fill_color' in point_mapping:
            style = {k: v for k, v in style.items() if k not in ['node_fill_color', 'node_nonselection_fill_color']}
            point_mapping['node_nonselection_fill_color'] = point_mapping['node_fill_color']
        edge_mapping = {}
        nan_node = index.max() + 1 if len(index) else 0
        start, end = (element.dimension_values(i) for i in range(2))
        if nodes.dtype.kind == 'f':
            start, end = (start.astype(np.int32), end.astype(np.int32))
        elif nodes.dtype.kind not in 'ui':
            start = np.array([node_indices.get(x, nan_node) for x in start], dtype=np.int32)
            end = np.array([node_indices.get(y, nan_node) for y in end], dtype=np.int32)
        path_data = dict(start=start, end=end)
        self._get_edge_colors(element, ranges, path_data, edge_mapping, style)
        if not static:
            pdata, pmapping = self._get_edge_paths(element, ranges)
            path_data.update(pdata)
            edge_mapping.update(pmapping)
        if 'hover' in self.handles:
            if self.inspection_policy == 'nodes':
                index_dim = element.nodes.get_dimension(2)
                point_data['index_hover'] = [index_dim.pprint_value(v) for v in element.nodes.dimension_values(2)]
                for d in element.nodes.dimensions()[3:]:
                    point_data[dimension_sanitizer(d.name)] = element.nodes.dimension_values(d)
            elif self.inspection_policy == 'edges':
                for d in element.dimensions():
                    dim_name = dimension_sanitizer(d.name)
                    if dim_name in ('start', 'end'):
                        dim_name += '_values'
                    path_data[dim_name] = element.dimension_values(d)
        data = {'scatter_1': point_data, self.edge_glyph: path_data, 'layout': layout}
        mapping = {'scatter_1': point_mapping, self.edge_glyph: edge_mapping}
        return (data, mapping, style)

    def _update_datasource(self, source, data):
        """
        Update datasource with data for a new frame.
        """
        if isinstance(source, ColumnDataSource):
            if self.handles['static_source']:
                source.trigger('data', source.data, data)
            else:
                source.data.update(data)
        else:
            source.graph_layout = data

    def _init_filled_edges(self, renderer, properties, edge_mapping):
        """Replace edge renderer with filled renderer"""
        glyph_model = Patches if self.filled else Bezier
        allowed_properties = glyph_model.properties()
        for glyph_type in ('', 'selection_', 'nonselection_', 'hover_', 'muted_'):
            glyph = getattr(renderer.edge_renderer, glyph_type + 'glyph', None)
            if glyph is None:
                continue
            group_properties = dict(properties)
            props = self._process_properties(self.edge_glyph, group_properties, {})
            filtered = self._filter_properties(props, glyph_type, allowed_properties)
            new_glyph = glyph_model(**dict(filtered, **edge_mapping))
            setattr(renderer.edge_renderer, glyph_type + 'glyph', new_glyph)

    def _get_graph_properties(self, plot, element, data, mapping, ranges, style):
        """Computes the args and kwargs for the GraphRenderer"""
        sources = []
        properties, mappings = ({}, {})
        for key in ('scatter_1', self.edge_glyph):
            gdata = data.pop(key, {})
            group_style = dict(style)
            style_group = self._style_groups.get('_'.join(key.split('_')[:-1]))
            with abbreviated_exception():
                group_style = self._apply_transforms(element, gdata, ranges, group_style, style_group)
            source = self._init_datasource(gdata)
            self.handles[key + '_source'] = source
            sources.append(source)
            others = [sg for sg in self._style_groups.values() if sg != style_group]
            glyph_props = self._glyph_properties(plot, element, source, ranges, group_style, style_group)
            for k, p in glyph_props.items():
                if any((k.startswith(o) for o in others)):
                    continue
                properties[k] = p
            mappings.update(mapping.pop(key, {}))
        properties = {p: v for p, v in properties.items() if p != 'source' and 'legend' not in p}
        properties.update(mappings)
        layout = data.pop('layout', {})
        layout = StaticLayoutProvider(graph_layout=layout)
        self.handles['layout_source'] = layout
        return (tuple(sources + [layout]), properties)

    def _reorder_renderers(self, plot, renderer, mapping):
        """Reorders renderers based on the defined draw order"""
        renderers = dict({r: self.handles[r + '_glyph_renderer'] for r in mapping}, graph=renderer)
        other = [r for r in plot.renderers if r not in renderers.values()]
        graph_renderers = [renderers[k] for k in self._draw_order if k in renderers]
        plot.renderers = other + graph_renderers

    def _set_interaction_policies(self, renderer):
        if self.selection_policy == 'nodes':
            renderer.selection_policy = NodesAndLinkedEdges()
        elif self.selection_policy == 'edges':
            renderer.selection_policy = EdgesAndLinkedNodes()
        else:
            renderer.selection_policy = NodesOnly()
        if self.inspection_policy == 'nodes':
            renderer.inspection_policy = NodesAndLinkedEdges()
        elif self.inspection_policy == 'edges':
            renderer.inspection_policy = EdgesAndLinkedNodes()
        else:
            renderer.inspection_policy = NodesOnly()

    def _init_glyphs(self, plot, element, ranges, source):
        style = self.style[self.cyclic_index]
        data, mapping, style = self.get_data(element, ranges, style)
        self.handles['previous_id'] = element._plot_id
        edge_mapping = {k: v for k, v in mapping[self.edge_glyph].items() if 'color' not in k}
        graph_args, properties = self._get_graph_properties(plot, element, data, mapping, ranges, style)
        renderer = plot.graph(*graph_args, **properties)
        if self.filled or self.bezier:
            self._init_filled_edges(renderer, properties, edge_mapping)
        self._set_interaction_policies(renderer)
        if data and mapping:
            CompositeElementPlot._init_glyphs(self, plot, element, ranges, source, data, mapping, style)
        if self._draw_order:
            self._reorder_renderers(plot, renderer, mapping)
        self.handles['glyph_renderer'] = renderer
        self.handles['scatter_1_glyph_renderer'] = renderer.node_renderer
        self.handles[self.edge_glyph + '_glyph_renderer'] = renderer.edge_renderer
        self.handles['scatter_1_glyph'] = renderer.node_renderer.glyph
        self.handles[self.edge_glyph + '_glyph'] = renderer.edge_renderer.glyph
        if 'hover' in self.handles:
            if self.handles['hover'].renderers == 'auto':
                self.handles['hover'].renderers = []
            self.handles['hover'].renderers.append(renderer)
        if self.colorbar:
            for k, v in list(self.handles.items()):
                if not k.endswith('color_mapper'):
                    continue
                self._draw_colorbar(plot, v, k.replace('color_mapper', ''))