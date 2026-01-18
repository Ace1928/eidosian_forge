from __future__ import annotations
import logging # isort:skip
from contextlib import contextmanager
from typing import (
import xyzservices
from ..core.enums import (
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.property_mixins import ScalarFillProps, ScalarLineProps
from ..core.query import find
from ..core.validation import error, warning
from ..core.validation.errors import (
from ..core.validation.warnings import MISSING_RENDERERS
from ..model import Model
from ..util.strings import nice_join
from ..util.warnings import warn
from .annotations import Annotation, Legend, Title
from .axes import Axis
from .dom import HTML
from .glyphs import Glyph
from .grids import Grid
from .layouts import GridCommon, LayoutDOM
from .ranges import (
from .renderers import GlyphRenderer, Renderer, TileRenderer
from .scales import (
from .sources import ColumnarDataSource, ColumnDataSource, DataSource
from .tiles import TileSource, WMTSTileSource
from .tools import HoverTool, Tool, Toolbar
class Plot(LayoutDOM):
    """ Model representing a plot, containing glyphs, guides, annotations.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def select(self, *args, **kwargs):
        """ Query this object and all of its references for objects that
        match the given selector.

        There are a few different ways to call the ``select`` method.
        The most general is to supply a JSON-like query dictionary as the
        single argument or as keyword arguments:

        Args:
            selector (JSON-like) : some sample text

        Keyword Arguments:
            kwargs : query dict key/values as keyword arguments

        Additionally, for compatibility with ``Model.select``, a selector
        dict may be passed as ``selector`` keyword argument, in which case
        the value of ``kwargs['selector']`` is used for the query.

        For convenience, queries on just names can be made by supplying
        the ``name`` string as the single parameter:

        Args:
            name (str) : the name to query on

        Also queries on just type can be made simply by supplying the
        ``Model`` subclass as the single parameter:

        Args:
            type (Model) : the type to query on

        Returns:
            seq[Model]

        Examples:

            .. code-block:: python

                # These three are equivalent
                p.select(selector={"type": HoverTool})
                p.select({"type": HoverTool})
                p.select(HoverTool)

                # These two are also equivalent
                p.select({"name": "mycircle"})
                p.select("mycircle")

                # Keyword arguments can be supplied in place of selector dict
                p.select({"name": "foo", "type": HoverTool})
                p.select(name="foo", type=HoverTool)

        """
        selector = _select_helper(args, kwargs)
        return _list_attr_splat(find(self.references(), selector))

    def row(self, row, gridplot):
        """ Return whether this plot is in a given row of a GridPlot.

        Args:
            row (int) : index of the row to test
            gridplot (GridPlot) : the GridPlot to check

        Returns:
            bool

        """
        return self in gridplot.row(row)

    def column(self, col, gridplot):
        """ Return whether this plot is in a given column of a GridPlot.

        Args:
            col (int) : index of the column to test
            gridplot (GridPlot) : the GridPlot to check

        Returns:
            bool

        """
        return self in gridplot.column(col)

    def _axis(self, *sides):
        objs = []
        for s in sides:
            objs.extend(getattr(self, s, []))
        axis = [obj for obj in objs if isinstance(obj, Axis)]
        return _list_attr_splat(axis)

    @property
    def xaxis(self):
        """ Splattable list of :class:`~bokeh.models.axes.Axis` objects for the x dimension.

        """
        return self._axis('above', 'below')

    @property
    def yaxis(self):
        """ Splattable list of :class:`~bokeh.models.axes.Axis` objects for the y dimension.

        """
        return self._axis('left', 'right')

    @property
    def axis(self):
        """ Splattable list of :class:`~bokeh.models.axes.Axis` objects.

        """
        return _list_attr_splat(self.xaxis + self.yaxis)

    @property
    def legend(self):
        """ Splattable list of |Legend| objects.

        """
        panels = self.above + self.below + self.left + self.right + self.center
        legends = [obj for obj in panels if isinstance(obj, Legend)]
        return _legend_attr_splat(legends)

    @property
    def hover(self):
        """ Splattable list of :class:`~bokeh.models.tools.HoverTool` objects.

        """
        hovers = [obj for obj in self.tools if isinstance(obj, HoverTool)]
        return _list_attr_splat(hovers)

    def _grid(self, dimension: Literal[0, 1]):
        grid = [obj for obj in self.center if isinstance(obj, Grid) and obj.dimension == dimension]
        return _list_attr_splat(grid)

    @property
    def xgrid(self):
        """ Splattable list of :class:`~bokeh.models.grids.Grid` objects for the x dimension.

        """
        return self._grid(0)

    @property
    def ygrid(self):
        """ Splattable list of :class:`~bokeh.models.grids.Grid` objects for the y dimension.

        """
        return self._grid(1)

    @property
    def grid(self):
        """ Splattable list of :class:`~bokeh.models.grids.Grid` objects.

        """
        return _list_attr_splat(self.xgrid + self.ygrid)

    @property
    def tools(self) -> list[Tool]:
        return self.toolbar.tools

    @tools.setter
    def tools(self, tools: list[Tool]):
        self.toolbar.tools = tools

    def add_layout(self, obj: Renderer, place: PlaceType='center') -> None:
        """ Adds an object to the plot in a specified place.

        Args:
            obj (Renderer) : the object to add to the Plot
            place (str, optional) : where to add the object (default: 'center')
                Valid places are: 'left', 'right', 'above', 'below', 'center'.

        Returns:
            None

        """
        if place not in Place:
            raise ValueError(f"Invalid place '{place}' specified. Valid place values are: {nice_join(Place)}")
        getattr(self, place).append(obj)

    def add_tools(self, *tools: Tool | str) -> None:
        """ Adds tools to the plot.

        Args:
            *tools (Tool) : the tools to add to the Plot

        Returns:
            None

        """
        for tool in tools:
            if isinstance(tool, str):
                tool_obj = Tool.from_string(tool)
            elif isinstance(tool, Tool):
                tool_obj = tool
            else:
                raise ValueError(f'expected a string or Tool instance, got {tool!r}')
            self.toolbar.tools.append(tool_obj)

    def remove_tools(self, *tools: Tool) -> None:
        """ Removes tools from the plot.

        Args:
            *tools (Tool) : the tools to remove from the Plot

        Returns:
            None

        """
        for tool in tools:
            if not isinstance(tool, Tool):
                raise ValueError('All arguments to remove_tool must be Tool subclasses.')
            elif tool not in self.toolbar.tools:
                raise ValueError(f'Invalid tool {tool} specified. Available tools are {nice_join(self.toolbar.tools)}')
            self.toolbar.tools.remove(tool)

    @overload
    def add_glyph(self, glyph: Glyph, **kwargs: Any) -> GlyphRenderer:
        ...

    @overload
    def add_glyph(self, source: ColumnarDataSource, glyph: Glyph, **kwargs: Any) -> GlyphRenderer:
        ...

    def add_glyph(self, source_or_glyph: Glyph | ColumnarDataSource, glyph: Glyph | None=None, **kwargs: Any) -> GlyphRenderer:
        """ Adds a glyph to the plot with associated data sources and ranges.

        This function will take care of creating and configuring a Glyph object,
        and then add it to the plot's list of renderers.

        Args:
            source (DataSource) : a data source for the glyphs to all use
            glyph (Glyph) : the glyph to add to the Plot


        Keyword Arguments:
            Any additional keyword arguments are passed on as-is to the
            Glyph initializer.

        Returns:
            GlyphRenderer

        """
        if isinstance(source_or_glyph, ColumnarDataSource):
            source = source_or_glyph
        else:
            source, glyph = (ColumnDataSource(), source_or_glyph)
        if not isinstance(source, DataSource):
            raise ValueError("'source' argument to add_glyph() must be DataSource subclass")
        if not isinstance(glyph, Glyph):
            raise ValueError("'glyph' argument to add_glyph() must be Glyph subclass")
        g = GlyphRenderer(data_source=source, glyph=glyph, **kwargs)
        self.renderers.append(g)
        return g

    def add_tile(self, tile_source: TileSource | xyzservices.TileProvider | str, retina: bool=False, **kwargs: Any) -> TileRenderer:
        """ Adds new ``TileRenderer`` into ``Plot.renderers``

        Args:
            tile_source (TileSource, xyzservices.TileProvider, str) :
                A tile source instance which contain tileset configuration

            retina (bool) :
                Whether to use retina version of tiles (if available)

        Keyword Arguments:
            Additional keyword arguments are passed on as-is to the tile renderer

        Returns:
            TileRenderer : TileRenderer

        """
        if not isinstance(tile_source, TileSource):
            if isinstance(tile_source, xyzservices.TileProvider):
                selected_provider = tile_source
            elif isinstance(tile_source, str):
                tile_source = tile_source.lower()
                if tile_source == 'esri_imagery':
                    tile_source = 'esri_worldimagery'
                if tile_source == 'osm':
                    tile_source = 'openstreetmap_mapnik'
                if tile_source.startswith('stamen'):
                    tile_source = f'stadia.{tile_source}'
                if 'retina' in tile_source:
                    tile_source = tile_source.replace('retina', '')
                    retina = True
                selected_provider = xyzservices.providers.query_name(tile_source)
            scale_factor = '@2x' if retina else None
            tile_source = WMTSTileSource(url=selected_provider.build_url(scale_factor=scale_factor), attribution=selected_provider.html_attribution, min_zoom=selected_provider.get('min_zoom', 0), max_zoom=selected_provider.get('max_zoom', 30))
        tile_renderer = TileRenderer(tile_source=tile_source, **kwargs)
        self.renderers.append(tile_renderer)
        return tile_renderer

    @contextmanager
    def hold(self, *, render: bool) -> Generator[None, None, None]:
        """ Takes care of turning a property on and off within a scope.

        Args:
            render (bool) :
                Turns the property hold_render on and off.
        """
        if render:
            self.hold_render = True
            yield
            self.hold_render = False

    @error(REQUIRED_RANGE)
    def _check_required_range(self) -> str | None:
        missing: list[str] = []
        if not self.x_range:
            missing.append('x_range')
        if not self.y_range:
            missing.append('y_range')
        if missing:
            return ', '.join(missing) + ' [%s]' % self

    @error(REQUIRED_SCALE)
    def _check_required_scale(self) -> str | None:
        missing: list[str] = []
        if not self.x_scale:
            missing.append('x_scale')
        if not self.y_scale:
            missing.append('y_scale')
        if missing:
            return ', '.join(missing) + ' [%s]' % self

    @error(INCOMPATIBLE_SCALE_AND_RANGE)
    def _check_compatible_scale_and_ranges(self) -> str | None:
        incompatible: list[str] = []
        x_ranges = list(self.extra_x_ranges.values())
        if self.x_range:
            x_ranges.append(self.x_range)
        y_ranges = list(self.extra_y_ranges.values())
        if self.y_range:
            y_ranges.append(self.y_range)
        if self.x_scale is not None:
            for rng in x_ranges:
                if isinstance(rng, (DataRange1d, Range1d)) and (not isinstance(self.x_scale, (LinearScale, LogScale))):
                    incompatible.append(f'incompatibility on x-dimension: {rng}, {self.x_scale}')
                elif isinstance(rng, FactorRange) and (not isinstance(self.x_scale, CategoricalScale)):
                    incompatible.append(f'incompatibility on x-dimension: {rng}, {self.x_scale}')
        if self.y_scale is not None:
            for rng in y_ranges:
                if isinstance(rng, (DataRange1d, Range1d)) and (not isinstance(self.y_scale, (LinearScale, LogScale))):
                    incompatible.append(f'incompatibility on y-dimension: {rng}, {self.y_scale}')
                elif isinstance(rng, FactorRange) and (not isinstance(self.y_scale, CategoricalScale)):
                    incompatible.append(f'incompatibility on y-dimension: {rng}, {self.y_scale}')
        if incompatible:
            return ', '.join(incompatible) + ' [%s]' % self

    @warning(MISSING_RENDERERS)
    def _check_missing_renderers(self) -> str | None:
        if len(self.renderers) == 0 and len([x for x in self.center if isinstance(x, Annotation)]) == 0:
            return str(self)

    @error(BAD_EXTRA_RANGE_NAME)
    def _check_bad_extra_range_name(self) -> str | None:
        msg: str = ''
        valid = {f'{axis}_name': {'default', *getattr(self, f'extra_{axis}s')} for axis in ('x_range', 'y_range')}
        for place in [*list(Place), 'renderers']:
            for ref in getattr(self, place):
                bad = ', '.join((f"{axis}='{getattr(ref, axis)}'" for axis, keys in valid.items() if getattr(ref, axis, 'default') not in keys))
                if bad:
                    msg += (', ' if msg else '') + f'{bad} [{ref}]'
        if msg:
            return msg
    x_range = Instance(Range, default=InstanceDefault(DataRange1d), help='\n    The (default) data range of the horizontal dimension of the plot.\n    ')
    y_range = Instance(Range, default=InstanceDefault(DataRange1d), help='\n    The (default) data range of the vertical dimension of the plot.\n    ')

    @classmethod
    def _scale(cls, scale: Literal['auto', 'linear', 'log', 'categorical']) -> Scale:
        if scale in ['auto', 'linear']:
            return LinearScale()
        elif scale == 'log':
            return LogScale()
        if scale == 'categorical':
            return CategoricalScale()
        else:
            raise ValueError(f'Unknown mapper_type: {scale}')
    x_scale = Instance(Scale, default=InstanceDefault(LinearScale), help='\n    What kind of scale to use to convert x-coordinates in data space\n    into x-coordinates in screen space.\n    ')
    y_scale = Instance(Scale, default=InstanceDefault(LinearScale), help='\n    What kind of scale to use to convert y-coordinates in data space\n    into y-coordinates in screen space.\n    ')
    extra_x_ranges = Dict(String, Instance(Range), help='\n    Additional named ranges to make available for mapping x-coordinates.\n\n    This is useful for adding additional axes.\n    ')
    extra_y_ranges = Dict(String, Instance(Range), help='\n    Additional named ranges to make available for mapping y-coordinates.\n\n    This is useful for adding additional axes.\n    ')
    extra_x_scales = Dict(String, Instance(Scale), help='\n    Additional named scales to make available for mapping x-coordinates.\n\n    This is useful for adding additional axes.\n\n    .. note:: This feature is experimental and may change in the short term.\n    ')
    extra_y_scales = Dict(String, Instance(Scale), help='\n    Additional named scales to make available for mapping y-coordinates.\n\n    This is useful for adding additional axes.\n\n    .. note:: This feature is experimental and may change in the short term.\n    ')
    hidpi = Bool(default=True, help='\n    Whether to use HiDPI mode when available.\n    ')
    title = Either(Null, Instance(Title), default=InstanceDefault(Title, text=''), help='\n    A title for the plot. Can be a text string or a Title annotation.\n    ').accepts(String, lambda text: Title(text=text))
    title_location = Nullable(Enum(Location), default='above', help='\n    Where the title will be located. Titles on the left or right side\n    will be rotated.\n    ')
    outline_props = Include(ScalarLineProps, prefix='outline', help='\n    The {prop} for the plot border outline.\n    ')
    outline_line_color = Override(default='#e5e5e5')
    renderers = List(Instance(Renderer), help='\n    A list of all glyph renderers for this plot.\n\n    This property can be manipulated by hand, but the ``add_glyph`` is\n    recommended to help make sure all necessary setup is performed.\n    ')
    toolbar = Instance(Toolbar, default=InstanceDefault(Toolbar), help='\n    The toolbar associated with this plot which holds all the tools. It is\n    automatically created with the plot if necessary.\n    ')
    toolbar_location = Nullable(Enum(Location), default='right', help='\n    Where the toolbar will be located. If set to None, no toolbar\n    will be attached to the plot.\n    ')
    toolbar_sticky = Bool(default=True, help='\n    Stick the toolbar to the edge of the plot. Default: True. If False,\n    the toolbar will be outside of the axes, titles etc.\n    ')
    toolbar_inner = Bool(default=False, help='\n    Locate the toolbar inside the frame. Setting this property to ``True``\n    makes most sense with auto-hidden toolbars.\n    ')
    left = List(Instance(Renderer), help='\n    A list of renderers to occupy the area to the left of the plot.\n    ')
    right = List(Instance(Renderer), help='\n    A list of renderers to occupy the area to the right of the plot.\n    ')
    above = List(Instance(Renderer), help='\n    A list of renderers to occupy the area above of the plot.\n    ')
    below = List(Instance(Renderer), help='\n    A list of renderers to occupy the area below of the plot.\n    ')
    center = List(Instance(Renderer), help='\n    A list of renderers to occupy the center area (frame) of the plot.\n    ')
    width: int | None = Override(default=600)
    height: int | None = Override(default=600)
    frame_width = Nullable(Int, help='\n    The width of a plot frame or the inner width of a plot, excluding any\n    axes, titles, border padding, etc.\n    ')
    frame_height = Nullable(Int, help='\n    The height of a plot frame or the inner height of a plot, excluding any\n    axes, titles, border padding, etc.\n    ')
    frame_align = Either(Bool, LRTB(Optional(Bool)), default=True, help='\n    Allows to specify which frame edges to align in multiple-plot layouts.\n\n    The default is to align all edges, but users can opt-out from alignment\n    of each individual edge or all edges. Note also that other properties\n    may disable alignment of certain edges, especially when using fixed frame\n    size (``frame_width`` and ``frame_height`` properties).\n    ')
    inner_width = Readonly(Int, help='\n    This is the exact width of the plotting canvas, i.e. the width of\n    the actual plot, without toolbars etc. Note this is computed in a\n    web browser, so this property will work only in backends capable of\n    bidirectional communication (server, notebook).\n\n    .. note::\n        This is an experimental feature and the API may change in near future.\n\n    ')
    inner_height = Readonly(Int, help='\n    This is the exact height of the plotting canvas, i.e. the height of\n    the actual plot, without toolbars etc. Note this is computed in a\n    web browser, so this property will work only in backends capable of\n    bidirectional communication (server, notebook).\n\n    .. note::\n        This is an experimental feature and the API may change in near future.\n\n    ')
    outer_width = Readonly(Int, help='\n    This is the exact width of the layout, i.e. the height of\n    the actual plot, with toolbars etc. Note this is computed in a\n    web browser, so this property will work only in backends capable of\n    bidirectional communication (server, notebook).\n\n    .. note::\n        This is an experimental feature and the API may change in near future.\n\n    ')
    outer_height = Readonly(Int, help='\n    This is the exact height of the layout, i.e. the height of\n    the actual plot, with toolbars etc. Note this is computed in a\n    web browser, so this property will work only in backends capable of\n    bidirectional communication (server, notebook).\n\n    .. note::\n        This is an experimental feature and the API may change in near future.\n\n    ')
    background_props = Include(ScalarFillProps, prefix='background', help='\n    The {prop} for the plot background style.\n    ')
    background_fill_color = Override(default='#ffffff')
    border_props = Include(ScalarFillProps, prefix='border', help='\n    The {prop} for the plot border style.\n    ')
    border_fill_color = Override(default='#ffffff')
    min_border_top = Nullable(Int, help='\n    Minimum size in pixels of the padding region above the top of the\n    central plot region.\n\n    .. note::\n        This is a *minimum*. The padding region may expand as needed to\n        accommodate titles or axes, etc.\n\n    ')
    min_border_bottom = Nullable(Int, help='\n    Minimum size in pixels of the padding region below the bottom of\n    the central plot region.\n\n    .. note::\n        This is a *minimum*. The padding region may expand as needed to\n        accommodate titles or axes, etc.\n\n    ')
    min_border_left = Nullable(Int, help='\n    Minimum size in pixels of the padding region to the left of\n    the central plot region.\n\n    .. note::\n        This is a *minimum*. The padding region may expand as needed to\n        accommodate titles or axes, etc.\n\n    ')
    min_border_right = Nullable(Int, help='\n    Minimum size in pixels of the padding region to the right of\n    the central plot region.\n\n    .. note::\n        This is a *minimum*. The padding region may expand as needed to\n        accommodate titles or axes, etc.\n\n    ')
    min_border = Nullable(Int, default=5, help='\n    A convenience property to set all all the ``min_border_X`` properties\n    to the same value. If an individual border property is explicitly set,\n    it will override ``min_border``.\n    ')
    lod_factor = Int(10, help='\n    Decimation factor to use when applying level-of-detail decimation.\n    ')
    lod_threshold = Nullable(Int, default=2000, help='\n    A number of data points, above which level-of-detail downsampling may\n    be performed by glyph renderers. Set to ``None`` to disable any\n    level-of-detail downsampling.\n    ')
    lod_interval = Int(300, help='\n    Interval (in ms) during which an interactive tool event will enable\n    level-of-detail downsampling.\n    ')
    lod_timeout = Int(500, help='\n    Timeout (in ms) for checking whether interactive tool events are still\n    occurring. Once level-of-detail mode is enabled, a check is made every\n    ``lod_timeout`` ms. If no interactive tool events have happened,\n    level-of-detail mode is disabled.\n    ')
    output_backend = Enum(OutputBackend, default='canvas', help='\n    Specify the output backend for the plot area. Default is HTML5 Canvas.\n\n    .. note::\n        When set to ``webgl``, glyphs without a WebGL rendering implementation\n        will fall back to rendering onto 2D canvas.\n    ')
    match_aspect = Bool(default=False, help='\n    Specify the aspect ratio behavior of the plot. Aspect ratio is defined as\n    the ratio of width over height. This property controls whether Bokeh should\n    attempt to match the (width/height) of *data space* to the (width/height)\n    in pixels of *screen space*.\n\n    Default is ``False`` which indicates that the *data* aspect ratio and the\n    *screen* aspect ratio vary independently. ``True`` indicates that the plot\n    aspect ratio of the axes will match the aspect ratio of the pixel extent\n    the axes. The end result is that a 1x1 area in data space is a square in\n    pixels, and conversely that a 1x1 pixel is a square in data units.\n\n    .. note::\n        This setting only takes effect when there are two dataranges. This\n        setting only sets the initial plot draw and subsequent resets. It is\n        possible for tools (single axis zoom, unconstrained box zoom) to\n        change the aspect ratio.\n\n    .. warning::\n        This setting is incompatible with linking dataranges across multiple\n        plots. Doing so may result in undefined behavior.\n    ')
    aspect_scale = Float(default=1, help='\n    A value to be given for increased aspect ratio control. This value is added\n    multiplicatively to the calculated value required for ``match_aspect``.\n    ``aspect_scale`` is defined as the ratio of width over height of the figure.\n\n    For example, a plot with ``aspect_scale`` value of 2 will result in a\n    square in *data units* to be drawn on the screen as a rectangle with a\n    pixel width twice as long as its pixel height.\n\n    .. note::\n        This setting only takes effect if ``match_aspect`` is set to ``True``.\n    ')
    reset_policy = Enum(ResetPolicy, default='standard', help='\n    How a plot should respond to being reset. By default, the standard actions\n    are to clear any tool state history, return plot ranges to their original\n    values, undo all selections, and emit a ``Reset`` event. If customization\n    is desired, this property may be set to ``"event_only"``, which will\n    suppress all of the actions except the Reset event.\n    ')
    hold_render = Bool(default=False, help="\n    When set to True all requests to repaint the plot will be hold off.\n\n    This is useful when periodically updating many glyphs. For example, let's\n    assume we have 10 lines on a plot, each with its own datasource. We stream\n    to all of them every second in a for loop like so:\n\n    .. code:: python\n\n        for line in lines:\n            line.stream(new_points())\n\n    The problem with this code is that every stream triggers a re-rendering of\n    the plot. Even tough repainting only on the last stream would produce almost\n    identical visual effect. Especially for lines with many points this becomes\n    computationally expensive and can freeze your browser. Using a convenience\n    method `hold`, we can control when rendering is initiated like so:\n\n    .. code:: python\n\n        with plot.hold(render=True):\n            for line in lines:\n                line.stream(new_points())\n\n    In this case we render newly appended points only after the last stream.\n    ")
    attribution = List(Either(Instance(HTML), String), default=[], help='\n    Allows to acknowledge or give credit to data, tile, etc. providers.\n\n    This can be in either HTML or plain text forms. Renderers, like\n    tile renderers, can provide additional attributions which will\n    be added after attributions provided here.\n\n    .. note::\n        This feature is experimental and may change in the short term.\n    ')